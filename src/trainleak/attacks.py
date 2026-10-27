"""Membership inference attacks.

Every attack maps model outputs to a *membership score*: higher means
"more likely this example was in the training set". Scores are only
compared against each other, so no attack needs a calibrated threshold.

References
----------
- Yeom et al., "Privacy Risk in Machine Learning", CSF 2018 (loss attack).
- Song & Mittal, "Systematic Evaluation of Privacy Risks of Machine
  Learning Models", USENIX Security 2021 (modified entropy).
- Carlini et al., "Membership Inference Attacks From First Principles",
  IEEE S&P 2022 (LiRA).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

EPS = 1e-12

ProbaFn = Callable[[object, np.ndarray], np.ndarray]
PredictFn = Callable[[object, np.ndarray], np.ndarray]
# LiRA statistic: one roughly Gaussian number per example from a fitted model.
StatisticFn = Callable[[object, np.ndarray, np.ndarray], np.ndarray]


def true_class_prob(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Probability the model assigns to each example's true label."""
    return probs[np.arange(len(labels)), labels]


def loss_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Negative cross-entropy loss (Yeom et al.). Members have lower loss."""
    return np.log(np.clip(true_class_prob(probs, labels), EPS, 1.0))


def confidence_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confidence in the true label."""
    return true_class_prob(probs, labels)


def entropy_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Negative prediction entropy. Label-free: members get sharper outputs."""
    p = np.clip(probs, EPS, 1.0)
    return np.sum(p * np.log(p), axis=1)


def modified_entropy_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Negative modified entropy (Song & Mittal 2021).

    Unlike plain entropy, a confident *wrong* prediction is scored as
    non-member-like, which makes this strictly more informative.
    """
    p = np.clip(probs, EPS, 1.0 - EPS)
    idx = np.arange(len(labels))
    p_y = p[idx, labels]
    term_true = (1.0 - p_y) * np.log(p_y)
    others = p * np.log(1.0 - p)
    others[idx, labels] = 0.0
    return term_true + others.sum(axis=1)


def squared_error_score(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Negative squared error, the regression analogue of the loss attack.

    A regressor fits its own training rows more closely than unseen ones,
    so members have smaller residuals (Yeom et al. 2018).
    """
    return -((preds - targets) ** 2)


def logit_scale(p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)): makes confidences approximately Gaussian (LiRA)."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


SIMPLE_ATTACKS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "loss": loss_score,
    "confidence": confidence_score,
    "entropy": entropy_score,
    "modified_entropy": modified_entropy_score,
}

REGRESSION_ATTACKS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "loss": squared_error_score,
}


def classification_statistic(predict_proba: ProbaFn) -> StatisticFn:
    """LiRA statistic for classifiers: logit-scaled confidence in the true label."""

    def fn(model, X, y):
        return logit_scale(true_class_prob(predict_proba(model, X), y))

    return fn


def regression_statistic(predict: PredictFn) -> StatisticFn:
    """LiRA statistic for regressors: the signed residual ``y - prediction``.

    Reference models fit their own rows tightly, so IN residuals sit in a
    narrower band around zero than OUT residuals. The likelihood ratio
    scores how much better the target model's residual fits the IN band.
    """

    def fn(model, X, y):
        return y - predict(model, X)

    return fn


def lira_scores(
    model_factory: Callable[[], object],
    statistic: StatisticFn,
    X: np.ndarray,
    y: np.ndarray,
    target_statistic: np.ndarray,
    *,
    n_reference: int = 16,
    fixed_variance: bool = True,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Online Likelihood Ratio Attack (Carlini et al. 2022).

    Trains ``n_reference`` reference models, each on a random half of the
    audit pool, so every example is IN for roughly half of them and OUT for
    the rest. For each example we fit a Gaussian to ``statistic`` of the IN
    models and of the OUT models, then score ``target_statistic`` (the same
    quantity from the model under audit) by the log likelihood ratio.

    ``statistic(model, X, y)`` returns one number per example that is
    roughly Gaussian across reference models: see ``classification_statistic``
    (logit-scaled confidence) and ``regression_statistic`` (residual).
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    phis = np.empty((n_reference, n))
    in_mask = np.zeros((n_reference, n), dtype=bool)

    for k in range(n_reference):
        idx = rng.random(n) < 0.5
        model = model_factory()
        model.fit(X[idx], y[idx])
        phis[k] = statistic(model, X, y)
        in_mask[k] = idx
        if progress:
            progress(k + 1, n_reference)

    def masked_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = np.maximum(mask.sum(axis=0), 1)
        mu = np.where(mask, phis, 0.0).sum(axis=0) / count
        var = np.where(mask, (phis - mu) ** 2, 0.0).sum(axis=0) / np.maximum(count - 1, 1)
        return mu, np.sqrt(var)

    mu_in, sd_in = masked_stats(in_mask)
    mu_out, sd_out = masked_stats(~in_mask)

    if fixed_variance:
        # With few reference models, per-example variances are noisy; the
        # paper recommends a single global variance in that regime.
        sd_in = np.full(n, np.median(sd_in))
        sd_out = np.full(n, np.median(sd_out))

    sd_in = np.maximum(sd_in, 1e-3)
    sd_out = np.maximum(sd_out, 1e-3)

    return norm.logpdf(target_statistic, mu_in, sd_in) - norm.logpdf(target_statistic, mu_out, sd_out)
