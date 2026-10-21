"""Metrics for membership inference.

Average-case metrics (AUC, accuracy) understate privacy risk: an attack
that confidently identifies 1% of members at near-zero false positives is
a real leak even if its AUC is 0.51. Following Carlini et al. (2022) we
report TPR at low FPR alongside AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_FPRS = (0.001, 0.01, 0.1)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """Highest TPR achievable while keeping FPR <= target."""
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


@dataclass
class AttackResult:
    name: str
    auc: float
    advantage: float
    accuracy: float
    tpr_at: dict[float, float]
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "advantage": round(self.advantage, 4),
            "accuracy": round(self.accuracy, 4),
            "tpr_at_fpr": {str(k): round(v, 4) for k, v in self.tpr_at.items()},
        }


def evaluate_attack(
    name: str, scores: np.ndarray, is_member: np.ndarray, fprs=DEFAULT_FPRS
) -> AttackResult:
    """Score an attack on a balanced set of members and non-members."""
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
    fpr, tpr, _ = roc_curve(is_member, scores)
    advantage = float(np.max(tpr - fpr))
    return AttackResult(
        name=name,
        auc=float(roc_auc_score(is_member, scores)),
        advantage=advantage,
        accuracy=0.5 + advantage / 2,
        tpr_at={f: tpr_at_fpr(fpr, tpr, f) for f in fprs},
        fpr=fpr,
        tpr=tpr,
        scores=scores,
    )
