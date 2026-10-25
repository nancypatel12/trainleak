"""The ``audit`` entry point and the ``AuditReport`` it returns."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .attacks import SIMPLE_ATTACKS, lira_scores, true_class_prob
from .metrics import AttackResult, evaluate_attack

ALL_ATTACKS = (*SIMPLE_ATTACKS, "lira")

RISK_LEVELS = ("low", "moderate", "high", "severe")


def risk_level(auc: float, tpr_at_1pct: float) -> str:
    """Heuristic grade from the strongest attack.

    The cut-offs are deliberately conservative: a 1% FPR attack that finds
    5x more members than chance is already "high".
    """
    if auc >= 0.80 or tpr_at_1pct >= 0.20:
        return "severe"
    if auc >= 0.65 or tpr_at_1pct >= 0.05:
        return "high"
    if auc >= 0.55 or tpr_at_1pct >= 0.02:
        return "moderate"
    return "low"


@dataclass
class AuditReport:
    results: dict[str, AttackResult]
    is_member: np.ndarray
    member_index: np.ndarray
    train_accuracy: float
    test_accuracy: float
    model_name: str = "model"

    @property
    def generalization_gap(self) -> float:
        return self.train_accuracy - self.test_accuracy

    @property
    def strongest(self) -> AttackResult:
        """Attack with the highest TPR at 1% FPR (ties broken by AUC)."""
        return max(self.results.values(), key=lambda r: (r.tpr_at.get(0.01, 0.0), r.auc))

    @property
    def risk(self) -> str:
        worst = max(RISK_LEVELS.index(risk_level(r.auc, r.tpr_at.get(0.01, 0.0))) for r in self.results.values())
        return RISK_LEVELS[worst]

    def most_vulnerable(self, k: int = 10, attack: str | None = None) -> np.ndarray:
        """Indices into ``X_train`` of the k members the attack is most sure about."""
        result = self.results[attack] if attack else self.strongest
        member_scores = result.scores[self.is_member == 1]
        order = np.argsort(-member_scores)[:k]
        return self.member_index[order]

    def summary(self) -> str:
        from .report import text_summary

        return text_summary(self)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "risk": self.risk,
            "strongest_attack": self.strongest.name,
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "n_members": int(self.is_member.sum()),
            "n_nonmembers": int((1 - self.is_member).sum()),
            "attacks": {name: r.to_dict() for name, r in self.results.items()},
        }

    def to_json(self, path: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def to_html(self, path: str) -> None:
        from .report import write_html

        write_html(self, path)

    def plot_roc(self, ax=None):
        from .report import plot_roc

        return plot_roc(self, ax=ax)

    def __repr__(self) -> str:
        s = self.strongest
        return (
            f"AuditReport(risk={self.risk!r}, strongest={s.name!r}, "
            f"auc={s.auc:.3f}, tpr@1%fpr={s.tpr_at.get(0.01, 0):.3f})"
        )


def _sklearn_proba(classes: np.ndarray) -> Callable[[object, np.ndarray], np.ndarray]:
    """predict_proba with columns aligned to ``classes`` even when a model
    was fit on a subset of them (common for LiRA reference models)."""

    def fn(model, X):
        raw = model.predict_proba(X)
        model_classes = getattr(model, "classes_", None)
        if model_classes is None or np.array_equal(model_classes, classes):
            return raw
        out = np.zeros((len(X), len(classes)))
        out[:, np.searchsorted(classes, model_classes)] = raw
        return out

    return fn


def _default_factory(model) -> Callable[[], object] | None:
    try:
        from sklearn.base import clone

        clone(model)
    except Exception:
        return None
    return lambda: clone(model)


def audit(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    attacks: Sequence[str] = ALL_ATTACKS,
    predict_proba: Callable[[object, np.ndarray], np.ndarray] | None = None,
    model_factory: Callable[[], object] | None = None,
    n_reference: int = 16,
    max_samples: int = 2000,
    seed: int = 0,
    verbose: bool = False,
) -> AuditReport:
    """Measure how much ``model`` leaks about its training set.

    Parameters
    ----------
    model
        A *fitted* classifier. Anything with ``predict_proba`` works; for
        other frameworks (PyTorch, JAX, ...) pass ``predict_proba``.
    X_train, y_train
        The data the model was trained on (the "members").
    X_test, y_test
        Held-out data from the same distribution (the "non-members").
    attacks
        Which attacks to run. ``"lira"`` retrains ``n_reference`` copies of
        the model, so it is the slowest and the strongest.
    predict_proba
        ``fn(model, X) -> probs`` with one column per class, in sorted label
        order. Defaults to ``model.predict_proba``.
    model_factory
        ``fn() -> unfitted model`` used by LiRA. Defaults to
        ``sklearn.base.clone(model)`` when possible.
    max_samples
        Cap on members and on non-members used for the audit (each).
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    rng = np.random.default_rng(seed)

    classes = np.unique(np.concatenate([y_train, y_test]))
    raw_proba = predict_proba or _sklearn_proba(classes)
    # LiRA trains reference models on encoded labels 0..k-1, not on the
    # original ones, so their columns align to those codes.
    raw_ref_proba = predict_proba or _sklearn_proba(np.arange(len(classes)))

    # float32 outputs (PyTorch, JAX) round 1 - 1e-12 to 1.0, which turns
    # clipped logs into -inf and silently breaks LiRA.
    def proba(m, X):
        return np.asarray(raw_proba(m, X), dtype=np.float64)

    def ref_proba(m, X):
        return np.asarray(raw_ref_proba(m, X), dtype=np.float64)

    enc_train = np.searchsorted(classes, y_train)
    enc_test = np.searchsorted(classes, y_test)

    # Balanced audit set: equal members and non-members, so accuracy and
    # advantage are meaningful and chance AUC is exactly 0.5.
    n = min(len(X_train), len(X_test), max_samples)
    mem_idx = rng.choice(len(X_train), n, replace=False)
    non_idx = rng.choice(len(X_test), n, replace=False)
    X = np.concatenate([X_train[mem_idx], X_test[non_idx]])
    y = np.concatenate([enc_train[mem_idx], enc_test[non_idx]])
    is_member = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    probs = proba(model, X)
    if probs.shape != (len(X), len(classes)):
        raise ValueError(f"predict_proba returned shape {probs.shape}, expected {(len(X), len(classes))}")
    preds = probs.argmax(axis=1)
    train_acc = float((preds[is_member == 1] == y[is_member == 1]).mean())
    test_acc = float((preds[is_member == 0] == y[is_member == 0]).mean())

    results: dict[str, AttackResult] = {}
    for name in attacks:
        if name in SIMPLE_ATTACKS:
            scores = SIMPLE_ATTACKS[name](probs, y)
        elif name == "lira":
            factory = model_factory or _default_factory(model)
            if factory is None:
                warnings.warn(
                    "Skipping LiRA: pass model_factory=lambda: <unfitted model> to enable it.",
                    stacklevel=2,
                )
                continue

            def progress(i, total):
                if verbose:
                    print(f"\r  lira: reference model {i}/{total}", end="", flush=True)

            scores = lira_scores(
                factory,
                ref_proba,
                X,
                y,
                true_class_prob(probs, y),
                n_reference=n_reference,
                seed=seed,
                progress=progress,
            )
            if verbose:
                print()
        else:
            raise ValueError(f"Unknown attack {name!r}. Choose from {ALL_ATTACKS}.")
        results[name] = evaluate_attack(name, scores, is_member)

    return AuditReport(
        results=results,
        is_member=is_member,
        member_index=mem_idx,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        model_name=type(model).__name__,
    )
