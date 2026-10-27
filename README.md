# trainleak

**Does your model leak its training data?**

[![CI](https://github.com/nancypatel12/trainleak/actions/workflows/ci.yml/badge.svg)](https://github.com/nancypatel12/trainleak/actions/workflows/ci.yml)

trainleak is an open-source Python library that checks whether a trained machine learning model leaks its training data.

Models can memorize the rows they were trained on. When they do, anyone who can query the model can often tell whether a specific person's record was used: a patient in a hospital's dataset, a customer in a bank's. This is called membership inference, and it's the standard way to measure leakage. trainleak runs the strongest published membership inference attacks against your model before you release it, and grades the risk from low to severe. Other kinds of leakage are on the [roadmap](#roadmap).

It works with scikit-learn models as is, and with PyTorch, JAX or any other framework through a small wrapper. It needs only NumPy, SciPy and scikit-learn, and runs on a CPU.

## Quickstart

### 1. Install

Python 3.9 or later.

```bash
pip install git+https://github.com/nancypatel12/trainleak
```

To also write HTML reports, install the `plot` extra instead (adds matplotlib):

```bash
pip install "trainleak[plot] @ git+https://github.com/nancypatel12/trainleak"
```

### 2. Run the demo

```bash
trainleak demo
```

The demo trains two models on scikit-learn's digits dataset, an overfit random forest and a regularized logistic regression, and audits both. It takes about 3 seconds on an M3 Pro.

### 3. Audit your own model

`audit` needs three things: the trained model, the data it was trained on, and data from the same source that it never saw. A held-out test split works.

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from trainleak import audit

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
report = audit(model, X_train, y_train, X_test, y_test)
print(report.summary())
```

```
trainleak audit: RandomForestClassifier
  members / non-members : 898 / 898
  train acc / test acc  : 1.000 / 0.973  (gap +0.027)

  attack                AUC    adv  TPR@0.1%   TPR@1%  TPR@10%
  loss                0.798  0.459     0.000    0.000    0.315
  confidence          0.798  0.459     0.000    0.000    0.315
  entropy             0.797  0.463     0.000    0.000    0.352
  modified_entropy    0.797  0.459     0.000    0.000    0.357
  lira                0.849  0.753     0.008    0.039    0.833

  risk: SEVERE  (strongest attack: lira)
  The model is memorizing its training set. Do not release it if the training data is sensitive.
```

### 4. Read the report

Rows the model trained on are "members". Rows it never saw are "non-members". Each attack tries to tell them apart.

| Field | Meaning |
|---|---|
| `risk` | The verdict: low, moderate, high or severe. Set by the strongest attack. |
| `AUC` | How well an attack separates members from non-members. 0.5 is a coin flip, 1.0 is perfect. |
| `TPR@1%` | The share of members an attack finds while wrongly flagging only 1% of non-members. The closest number to "how many real people are exposed." `TPR@0.1%` and `TPR@10%` are the same at stricter and looser cutoffs. |
| `adv` | Attack advantage: the largest gap between true-positive and false-positive rate. 0 means no better than guessing. |

A low risk means these attacks failed, not that the model is private. See [Limitations](#limitations).

## Usage

### Arguments

Everything after the five data arguments is optional.

| Argument | What to pass |
|---|---|
| `model` | A fitted model. Classifiers need `predict_proba`, regressors need `predict`. |
| `X_train`, `y_train` | The rows the model was trained on. |
| `X_test`, `y_test` | Rows from the same source that the model never saw. |
| `task` | `"classification"` (default) or `"regression"`. |
| `attacks` | Which attacks to run, e.g. `["loss", "lira"]`. Defaults to every attack that applies to the task. |
| `n_reference` | How many reference models LiRA trains (default 16). More is slower but more reliable. |
| `model_factory` | A function that returns a fresh, unfitted model. LiRA needs it for models scikit-learn can't copy. |
| `predict_proba`, `predict` | `fn(model, X)` to use in place of the model's own methods. |
| `max_samples` | The most rows used from each side (default 2,000). |
| `seed` | Random seed (default 0). |

### Regression models

Models that predict a number, such as a price or a dose, leak too: their errors are smaller on rows they trained on. Pass `task="regression"`. The report shows mean squared error in place of accuracy and runs the two attacks that apply, `loss` and `lira`.

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from trainleak import audit

X, y = make_regression(n_samples=800, n_features=20, n_informative=5, noise=10.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
report = audit(model, X_train, y_train, X_test, y_test, task="regression")
print(report.summary())
```

```
trainleak audit: RandomForestRegressor
  members / non-members : 400 / 400
  train mse / test mse  : 118.4 / 1128  (gap +1009)

  attack                AUC    adv  TPR@0.1%   TPR@1%  TPR@10%
  loss                0.789  0.460     0.018    0.062    0.360
  lira                0.881  0.625     0.142    0.170    0.510

  risk: SEVERE  (strongest attack: lira)
  The model is memorizing its training set. Do not release it if the training data is sensitive.
```

### PyTorch and other frameworks

Wrap your model in a class with `fit(X, y)` and `predict_proba(X)`. LiRA calls `model_factory` to train fresh copies. Without it, LiRA is skipped with a warning and the other attacks still run. This example uses the digits data from the Quickstart.

```python
import torch
from torch import nn
from trainleak import audit

class Net:
    def __init__(self, d, k):
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, k))

    def fit(self, X, y, epochs=200):
        X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(self.net(X), y).backward()
            opt.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).softmax(-1).numpy()

make = lambda: Net(X_train.shape[1], 10)   # labels must be 0..k-1
model = make().fit(X_train, y_train)
report = audit(model, X_train, y_train, X_test, y_test, model_factory=make, n_reference=8)
```

If you can't wrap the model, pass `predict_proba=lambda model, X: ...` for a classifier or `predict=lambda model, X: ...` for a regressor.

### Work with the results

```python
report.risk                     # "low" | "moderate" | "high" | "severe"
report.strongest.tpr_at[0.01]   # TPR@1% of the strongest attack
report.most_vulnerable(10)      # indices into X_train of the 10 most exposed rows
report.to_json("audit.json")    # every number, for scripts and CI
report.to_html("audit.html")    # a standalone page with ROC curves (needs the plot extra)
```

### Fail a CI build when a model leaks

```python
report = audit(model, X_train, y_train, X_test, y_test)
assert report.risk in ("low", "moderate"), report.summary()
```

## How it works

Every attack asks the same question about each row: does the model treat it like something it has already seen? The simple attacks read one number from the model's output. LiRA goes further: it trains copies of your model with and without each row, learns what "seen" and "unseen" look like for that row, and compares your model against both.

| Attack | Idea | Cost |
|---|---|---|
| `loss` | Members have lower loss (Yeom et al., 2018) | free |
| `confidence` | Members get higher confidence on their true label | free |
| `entropy` | Members get sharper predictions; needs no labels | free |
| `modified_entropy` | Entropy that penalizes confident *wrong* predictions (Song & Mittal, 2021) | free |
| `lira` | Train *N* reference models on random halves, fit per-example Gaussians to logit-scaled confidence IN vs OUT, score by likelihood ratio (Carlini et al., 2022) | *N* fits |

For regression, `loss` is negative squared error and `lira` fits its Gaussians to the signed residual `y - prediction`. The entropy and confidence attacks need class probabilities, so they don't apply.

The audit set is balanced, so chance is exactly AUC 0.5. Risk is the worst grade any attack earns: **severe** at AUC ≥ 0.80 or TPR@1% ≥ 20%, **high** at 0.65 / 5%, **moderate** at 0.55 / 2%, else **low**. These thresholds are conservative heuristics, not guarantees.

## Benchmarks

The benchmarks audit five classifiers on three datasets: 15 models, 16 LiRA reference models each, 50/50 splits, one seed. Three findings stand out. The table shows the rows behind each one.

| Finding | Dataset | Model | Test acc | TPR@1%, LiRA | TPR@1%, loss | Risk |
|---|---|---|---|---|---|---|
| 1 | digits | GradBoost | 0.949 | 0.186 | 0.012 | **high** |
| 1 | synthetic-4class | MLP (no reg.) | 0.686 | 0.391 | 0.011 | **severe** |
| 2 | digits | MLP (no reg.) | 0.973 | 0.081 | 0.004 | **high** |
| 2 | digits | MLP (L2 + early stop) | 0.937 | 0.016 | 0.008 | **low** |
| 3 | synthetic-4class | RandomForest | 0.729 | 0.989 | 0.261 | **severe** |

1. **Simple attacks underestimate risk.** At 1% FPR, LiRA finds at least 10× more members than the loss attack in 9 of the 15 models. In 3 of them, the loss attack finds none.
2. **Regularization is a privacy control.** L2 and early stopping take the digits MLP from *high* to *low* risk for 3.6 points of accuracy.
3. **Accuracy doesn't reveal leakage.** The synthetic random forest scores only 73% on test data, yet LiRA finds 99% of its training rows while flagging just 1% of unseen ones.

All 15 results: [`benchmarks/results.md`](benchmarks/results.md). Reproduce them with `python benchmarks/run_benchmarks.py` (2 minutes on an M3 Pro CPU).

## Design decisions

- **TPR at low FPR, not just AUC.** Confidently exposing 1% of your users is a leak, even at AUC 0.51 (Carlini et al., 2022).
- **A balanced audit set.** Equal members and non-members, so chance is exactly AUC 0.5 and advantage means something.
- **Online LiRA with one global variance.** With 16 reference models, per-example variances are noisy; the paper recommends a fixed variance in that regime.
- **The worst attack sets the risk.** One attack that works is enough to leak.
- **No deep learning dependency.** PyTorch, JAX and others plug in through `predict_proba` and `model_factory`, so the core stays NumPy, SciPy and scikit-learn.

## Limitations

- A low score means these attacks failed, not that the model is private. Only differential privacy gives worst-case guarantees.
- LiRA's reference models train on half the audit pool (up to 2,000 rows by default), not your full training set. On large datasets, raise `max_samples` so they behave more like your model.
- LiRA retrains your model `n_reference` times. That's cheap for classical ML. For large networks, lower it or skip LiRA.
- Benchmarks use one seed, so small differences may be noise.
- Membership inference only, on classifiers and regressors. Other leakage attacks and generative models are on the [roadmap](#roadmap).

## Roadmap

The goal is for trainleak to measure every well-established way a model leaks its training data, behind one call, with each attack tested against the results in its paper and graded with a clear risk level. Membership inference came first because it's the standard measure of leakage. The rest follows in this order:

1. **More membership attacks.** Label-only attacks, which need only the predicted label.
2. **Attribute inference.** Recovering a hidden field of a known person, such as income, from the model.
3. **Training data extraction.** Pulling verbatim training text or images out of generative models.

Alongside the attacks: adapters for more frameworks, starting with Hugging Face transformers, and a pytest plugin that fails a test when a model leaks.

Out of scope: adversarial examples, model stealing and data poisoning. They are about model security, not training data leakage, and other toolkits cover them.

Open work is tracked in [issues](https://github.com/nancypatel12/trainleak/issues).

## Contributing

Contributions are welcome, and new attacks most of all.

- **Adding an attack.** Open an issue first and link the paper. The steps are in [CONTRIBUTING.md](CONTRIBUTING.md#add-an-attack).
- **Finding a first task.** Issues labeled [good first issue](https://github.com/nancypatel12/trainleak/labels/good%20first%20issue) are small and say where in the code to start.
- **Reporting a bug.** Open an issue with the bug report form.

Setup steps and the pull request checklist are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
