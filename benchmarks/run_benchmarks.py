"""Reproduce the benchmark table and figure in the README.

    python benchmarks/run_benchmarks.py

Runs in about a minute on a laptop CPU. Writes benchmarks/results.md,
benchmarks/results.json and docs/roc.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    rows, figure_reports = [], []
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
                    "risk": r.risk,
                    "seconds": round(time.time() - start, 1),
                }
            )
            print(rows[-1])
            if ds_name == "digits" and m_name in ("RandomForest", "LogReg (C=0.05)"):
                r.model_name = m_name
                figure_reports.append(r)

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

    fig, axes = plt.subplots(1, len(figure_reports), figsize=(11, 5))
    for ax, r in zip(axes, figure_reports):
        r.plot_roc(ax=ax)
        ax.set_title(f"{r.model_name}: {r.risk.upper()} risk\n(test acc {r.test_accuracy:.3f})")
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "roc.png", dpi=140)
    print(header + body)


if __name__ == "__main__":
    main()
