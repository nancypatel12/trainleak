# Contributing to trainleak

Thanks for helping make ML models safer to release. Contributions of every size are welcome: a typo fix, a new attack, a benchmark on a dataset you care about.

## Where to start

- Issues labeled [`good first issue`](https://github.com/nancypatel12/trainleak/labels/good%20first%20issue) are scoped to a few hours and have pointers to the relevant code.
- Issues labeled [`help wanted`](https://github.com/nancypatel12/trainleak/labels/help%20wanted) are larger and open to anyone.
- Have an idea that isn't in an issue yet? Open one first so we can agree on the approach before you write code.

Comment on an issue to claim it. If there's no PR or update after 14 days, it's open again.

## Development setup

```bash
git clone https://github.com/<you>/trainleak && cd trainleak
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # all tests should pass in a few seconds
ruff check .      # lint
```

## Pull request checklist

1. Branch from `main`: `git checkout -b feat/short-description`.
2. Keep each PR to one change. Small PRs are reviewed faster.
3. Add or update tests. New attacks need a test showing they beat chance on an overfit model.
4. `pytest` and `ruff check .` pass locally (CI runs both on Python 3.9 to 3.13).
5. Update `README.md` if you change the public API or add an attack.
6. Fill in the PR template and link the issue (`Closes #12`).

## Adding a new attack

Attacks live in `src/trainleak/attacks.py`. A score-based attack is a function
`(probs, labels) -> scores` where a higher score means "more likely a member".
Register it in `SIMPLE_ATTACKS`, cite the paper in the docstring, and add a row
to the benchmark by running `python benchmarks/run_benchmarks.py`.

## Style

- Follow the existing code: type hints, NumPy-style docstrings, no new
  required dependencies without discussion (optional extras are fine).
- Commit messages: imperative mood, e.g. `Add label-only attack`.

## Reporting security issues

trainleak is a measurement tool, but if you find a bug that makes it report a
model as safer than it is, please treat that as high priority and open an issue
with the `bug` label.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind
and assume good intent.
