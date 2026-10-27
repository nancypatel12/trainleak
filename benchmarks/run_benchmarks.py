"""Reproduce the benchmark results in the README.

    python benchmarks/run_benchmarks.py

Runs in about 2 minutes on an M3 Pro CPU, one seed. Writes benchmarks/results.md
and benchmarks/results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trainleak import audit

ROOT = Path(__file__).resolve().parent.parent


def datasets():
    yield "digits", *load_digits(return_X_y=True)
    yield "breast_cancer", *load_breast_cancer(return_X_y=True)
    X, y = make_classification(n_samples=3000, n_features=40, n_informative=8, n_classes=4, flip_y=0.05, random_state=0)
    yield "synthetic-4class", X, y


def models():
    yield "LogReg (C=0.05)", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    yield "RandomForest", lambda: RandomForestClassifier(n_estimators=100, random_state=0)
    yield "GradBoost", lambda: GradientBoostingClassifier(random_state=0)
    yield (
        "MLP (no reg.)",
        lambda: make_pipeline(StandardScaler(), MLPClassifier((256, 256), alpha=0.0, max_iter=500, random_state=0)),
    )
    yield (
        "MLP (L2 + early stop)",
        lambda: make_pipeline(
            StandardScaler(), MLPClassifier((256, 256), alpha=1.0, early_stopping=True, random_state=0)
        ),
    )


def main() -> None:
    rows = []
    for ds_name, X, y in datasets():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
        for m_name, factory in models():
            start = time.time()
            model = factory().fit(X_tr, y_tr)
            r = audit(model, X_tr, y_tr, X_te, y_te, model_factory=factory, n_reference=16)
            s = r.strongest
            rows.append(
                {
                    "dataset": ds_name,
                    "model": m_name,
                    "test_acc": round(r.test_accuracy, 3),
                    "gap": round(r.generalization_gap, 3),
                    "best_attack": s.name,
                    "auc": round(s.auc, 3),
                    "tpr@0.1%": round(s.tpr_at[0.001], 3),
                    "tpr@1%": round(s.tpr_at[0.01], 3),
                    "loss_tpr@0.1%": round(r.results["loss"].tpr_at[0.001], 3),
                    "loss_tpr@1%": round(r.results["loss"].tpr_at[0.01], 3),
                    "risk": r.risk,
                    "seconds": round(time.time() - start, 1),
                }
            )
            print(rows[-1])

    out = ROOT / "benchmarks"
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    header = "| dataset | model | test acc | train-test gap | best attack | AUC | TPR@0.1%FPR | TPR@1%FPR | risk |\n"
    header += "|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['dataset']} | {r['model']} | {r['test_acc']:.3f} | {r['gap']:+.3f} | {r['best_attack']} | "
        f"{r['auc']:.3f} | {r['tpr@0.1%']:.3f} | {r['tpr@1%']:.3f} | **{r['risk']}** |\n"
        for r in rows
    )
    (out / "results.md").write_text(header + body)
    total = sum(row["seconds"] for row in rows)
    print(f"{len(rows)} audits in {total:.0f} s")
    print(header + body)


if __name__ == "__main__":
    main()
