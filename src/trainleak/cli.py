"""Command line interface: ``trainleak demo``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import __version__, audit


def demo(args: argparse.Namespace) -> None:
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

    models = {
        "overfit random forest": RandomForestClassifier(n_estimators=100, random_state=0),
        "regularized logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
    }
    print("Dataset: sklearn digits (898 train / 899 test). Two models, same data.\n")
    for i, (label, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        start = time.time()
        report = audit(model, X_train, y_train, X_test, y_test, n_reference=args.n_reference, verbose=True)
        print(f"[{label}]  ({time.time() - start:.1f}s)")
        print(report.summary(), "\n")
        if args.html:
            base = Path(args.html)
            path = str(base.with_name(f"{base.stem}-{i}{base.suffix or '.html'}"))
            report.to_html(path)
            print(f"  HTML report: {path}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trainleak", description="Membership inference audits for ML models.")
    parser.add_argument("--version", action="version", version=f"trainleak {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="audit an overfit and a regularized model side by side")
    d.add_argument("--n-reference", type=int, default=16, help="LiRA reference models (default 16)")
    d.add_argument("--html", help="also write HTML reports, e.g. report.html")
    d.set_defaults(func=demo)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
