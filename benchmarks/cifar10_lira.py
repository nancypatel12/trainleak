"""Audit a PyTorch CNN trained on CIFAR-10, the setting of the LiRA paper.

    pip install torch
    python benchmarks/cifar10_lira.py                     # full run: 10,000 members, 16 reference models
    python benchmarks/cifar10_lira.py --smoke             # 500 members, 2 reference models, 2 epochs

trainleak trains the reference models itself, so the only PyTorch-specific code
is the wrapper below: fit(X, y) and predict_proba(X). Writes
benchmarks/cifar10_results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import platform
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trainleak import audit

ROOT = Path(__file__).resolve().parent.parent

# Per-channel mean and std of the CIFAR-10 training set.
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


class SmallCNN(nn.Module):
    """Four conv blocks and a linear head, about 1.2M parameters."""

    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(c_in: int, c_out: int, pool: bool) -> nn.Sequential:
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 64, pool=False),
            block(64, 128, pool=True),  # 32 -> 16
            block(128, 256, pool=True),  # 16 -> 8
            block(256, 256, pool=True),  # 8 -> 4
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop with 4-pixel padding and random horizontal flip, as in the LiRA paper."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    # Each image gets its own crop offset: index rows and columns per image.
    offsets = torch.randint(0, 9, (n, 2), device=x.device)
    span = torch.arange(32, device=x.device)
    rows = (offsets[:, 0:1] + span).view(n, 1, 32, 1)
    cols = (offsets[:, 1:2] + span).view(n, 1, 1, 32)
    batch = torch.arange(n, device=x.device).view(n, 1, 1, 1)
    channel = torch.arange(3, device=x.device).view(1, 3, 1, 1)
    return padded[batch, channel, rows, cols]


class TorchClassifier:
    """The two methods trainleak needs: fit(X, y) and predict_proba(X).

    X is a uint8 array of images, shape (n, 32, 32, 3). Each instance trains a
    fresh SmallCNN, so trainleak can build reference models from a factory.
    """

    def __init__(self, epochs: int, device: str, seed: int, batch_size: int = 256, lr: float = 0.1):
        self.epochs, self.device, self.seed = epochs, device, seed
        self.batch_size, self.lr = batch_size, lr
        self.net: SmallCNN | None = None

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float().div(255)
        return ((x - MEAN) / STD).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchClassifier:
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        x_all, y_all = self._to_tensor(X), torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.net = SmallCNN().to(self.device)
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        steps = self.epochs * -(-len(X) // self.batch_size)
        # One-cycle schedule: trains a small CNN well in few epochs.
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(X), generator=g).to(self.device)
            for i in range(0, len(X), self.batch_size):
                idx = order[i : i + self.batch_size]
                loss = F.cross_entropy(self.net(augment(x_all[idx])), y_all[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = [F.softmax(self.net(self._to_tensor(X[i : i + 1024])), dim=1).cpu() for i in range(0, len(X), 1024)]
        # float64, because float32 rounds confident outputs to exactly 1.0.
        return torch.cat(out).double().numpy()


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """The 50,000 CIFAR-10 training images as uint8 (n, 32, 32, 3), and their labels."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != CIFAR10_MD5:
        raise RuntimeError(f"{archive} is corrupt or incomplete; delete it and run again")
    images, labels = [], []
    with tarfile.open(archive) as tar:
        for i in range(1, 6):
            batch = pickle.load(tar.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(batch[b"labels"])
    return np.concatenate(images), np.asarray(labels)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", type=int, default=10_000, help="training set size of the audited model")
    parser.add_argument("--n-reference", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "cifar10_results.json")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the plumbing")
    args = parser.parse_args()
    if args.smoke:
        args.members, args.n_reference, args.epochs = 500, 2, 2

    device = pick_device(args.device)
    X, y = load_cifar10(args.data)
    # Members and non-members come from the same 50,000 training images, split at random.
    order = np.random.default_rng(args.seed).permutation(len(X))
    mem, non = order[: args.members], order[args.members : 2 * args.members]

    # Each model gets its own seed: the target is seed + 0, reference model k is seed + k.
    seeds = itertools.count(args.seed)

    def factory() -> TorchClassifier:
        return TorchClassifier(args.epochs, device, next(seeds))

    start = time.time()
    model = factory().fit(X[mem], y[mem])
    target_seconds = time.time() - start
    print(f"target model trained in {target_seconds:.0f} s on {device}")

    report = audit(
        model,
        X[mem],
        y[mem],
        X[non],
        y[non],
        model_factory=factory,
        n_reference=args.n_reference,
        max_samples=args.members,
        seed=args.seed,
        verbose=True,
    )
    print(report.summary())

    result = {
        "config": {
            "members": args.members,
            "non_members": args.members,
            "n_reference": args.n_reference,
            "epochs": args.epochs,
            "seed": args.seed,
            "smoke": args.smoke,
        },
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.machine(),
            "torch": torch.__version__,
        },
        "train_accuracy": round(report.train_accuracy, 4),
        "test_accuracy": round(report.test_accuracy, 4),
        "attacks": {name: r.to_dict() for name, r in report.results.items()},
        "risk": report.risk,
        "seconds": {"target": round(target_seconds), "total": round(time.time() - start)},
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
