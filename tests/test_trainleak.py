import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from trainleak import ALL_ATTACKS, audit
from trainleak.attacks import entropy_score, logit_scale, loss_score, modified_entropy_score
from trainleak.metrics import evaluate_attack, tpr_at_fpr


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=800, n_features=20, n_informative=5, flip_y=0.1, random_state=0)
    return train_test_split(X, y, test_size=0.5, random_state=0)


def test_scores_rank_confident_correct_above_wrong():
    probs = np.array([[0.99, 0.01], [0.6, 0.4], [0.1, 0.9]])
    labels = np.array([0, 0, 0])
    for fn in (loss_score, modified_entropy_score):
        s = fn(probs, labels)
        assert s[0] > s[1] > s[2]


def test_entropy_is_label_free():
    probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    assert np.allclose(entropy_score(probs, np.array([0, 0])), entropy_score(probs, np.array([1, 1])))


def test_logit_scale_is_finite_at_extremes():
    assert np.isfinite(logit_scale(np.array([0.0, 1.0]))).all()


def test_perfect_and_random_attacks():
    is_member = np.array([1] * 100 + [0] * 100)
    perfect = evaluate_attack("p", is_member.astype(float), is_member)
    assert perfect.auc == 1.0 and perfect.tpr_at[0.001] == 1.0

    rng = np.random.default_rng(0)
    random = evaluate_attack("r", rng.random(200), is_member)
    assert 0.35 < random.auc < 0.65


def test_tpr_at_fpr_respects_bound():
    fpr = np.array([0.0, 0.005, 0.02, 1.0])
    tpr = np.array([0.0, 0.3, 0.6, 1.0])
    assert tpr_at_fpr(fpr, tpr, 0.01) == 0.3


def test_overfit_model_leaks_more_than_regularized(data):
    X_tr, X_te, y_tr, y_te = data
    overfit = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    regular = LogisticRegression(C=0.01).fit(X_tr, y_tr)
    r_over = audit(overfit, X_tr, y_tr, X_te, y_te, n_reference=8)
    r_reg = audit(regular, X_tr, y_tr, X_te, y_te, n_reference=8)
    assert r_over.strongest.auc > r_reg.strongest.auc + 0.1
    assert r_over.risk in ("high", "severe")


def test_report_contents(data, tmp_path):
    X_tr, X_te, y_tr, y_te = data
    model = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    report = audit(model, X_tr, y_tr, X_te, y_te, n_reference=4, max_samples=200)

    assert set(report.results) == set(ALL_ATTACKS)
    assert report.is_member.sum() == 200
    vuln = report.most_vulnerable(5)
    assert len(vuln) == 5 and vuln.max() < len(X_tr)

    d = json.loads(report.to_json(str(tmp_path / "r.json")))
    assert d["risk"] == report.risk
    assert "TPR@1%" in report.summary()

    report.to_html(str(tmp_path / "r.html"))
    assert "data:image/png;base64," in (tmp_path / "r.html").read_text()


def test_custom_predict_proba_without_sklearn(data):
    X_tr, X_te, y_tr, y_te = data

    class Wrapped:  # e.g. a PyTorch module behind a small adapter
        def __init__(self, inner):
            self.inner = inner

    inner = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    with pytest.warns(UserWarning, match="Skipping LiRA"):
        report = audit(Wrapped(inner), X_tr, y_tr, X_te, y_te, predict_proba=lambda m, X: m.inner.predict_proba(X))
    assert "lira" not in report.results and report.strongest.auc > 0.6


def test_float32_probabilities_keep_lira_working(data):
    X_tr, X_te, y_tr, y_te = data
    model = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    report = audit(
        model, X_tr, y_tr, X_te, y_te,
        predict_proba=lambda m, X: m.predict_proba(X).astype(np.float32),
        model_factory=lambda: DecisionTreeClassifier(random_state=0),
        n_reference=8,
    )
    assert report.results["lira"].auc > 0.6


def test_lira_ignores_how_labels_are_named(data):
    X_tr, X_te, y_tr, y_te = data
    aucs = []
    for names in (np.array([0, 1]), np.array([-1, 1]), np.array(["cat", "dog"])):
        model = DecisionTreeClassifier(random_state=0).fit(X_tr, names[y_tr])
        report = audit(model, X_tr, names[y_tr], X_te, names[y_te], attacks=["lira"], n_reference=8)
        aucs.append(report.results["lira"].auc)
    assert aucs[0] > 0.6
    assert np.allclose(aucs, aucs[0])


def test_string_labels_and_unknown_attack(data):
    X_tr, X_te, y_tr, y_te = data
    names = np.array(["cat", "dog"])
    model = LogisticRegression().fit(X_tr, names[y_tr])
    report = audit(model, X_tr, names[y_tr], X_te, names[y_te], attacks=["loss"])
    assert list(report.results) == ["loss"]
    with pytest.raises(ValueError, match="Unknown attack"):
        audit(model, X_tr, names[y_tr], X_te, names[y_te], attacks=["nope"])
