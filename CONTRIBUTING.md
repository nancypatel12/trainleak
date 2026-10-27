# Contributing to trainleak

Thanks for helping make trained models safer to release. Any contribution helps, whether it fixes a typo or adds a new attack.

## Ways to contribute

| You want to | Do this |
|---|---|
| Fix a typo or a small bug yourself | Open a pull request. |
| Add an attack or a feature | Open an issue with the feature request form first, so we agree on the approach before you write code. |
| Report a bug | Open an issue with the bug report form. |
| Ask a question | Post it in [Discussions](https://github.com/nancypatel12/trainleak/discussions). |

## Find something to work on

1. Pick an open issue. Issues labeled [`good first issue`](https://github.com/nancypatel12/trainleak/labels/good%20first%20issue) are small and say where in the code to start. Issues labeled [`help wanted`](https://github.com/nancypatel12/trainleak/labels/help%20wanted) are larger.
2. Comment on the issue to say you're working on it.
3. If you haven't opened a pull request or posted an update within 14 days, someone else can take it.

## Set up

Fork the repository on GitHub, then run:

```bash
git clone https://github.com/<your-username>/trainleak
cd trainleak
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs trainleak from your copy of the code, along with the tools for testing and checking it.

## Make your change

- Add or update tests in `tests/test_trainleak.py` for anything that changes how trainleak behaves.
- Give new functions type hints and a docstring written like the existing ones.
- Ask in an issue before adding a new required dependency. Optional extras are fine.
- If you change how people call trainleak, or add an attack, update `README.md`.

## Add an attack

New attacks are the most valuable contribution. The goal is for trainleak to cover every well-established way a model leaks its training data (see the [roadmap](README.md#roadmap)).

An attack gives each row a score: how likely it is that the row was in the training data. Higher means more likely.

1. **Open an issue first.** Link the paper and explain what the attack catches that the current attacks miss.
2. **Write the attack** as a function in `src/trainleak/attacks.py`, and cite the paper in its docstring.
   - For classifiers, the function takes `probs` (the model's predicted probabilities, one column per class) and `labels`, and returns one score per row. Add it to the `SIMPLE_ATTACKS` dictionary.
   - For regressors, it takes `preds` (the model's predictions) and `targets`. Add it to the `REGRESSION_ATTACKS` dictionary.
3. **Test it.** Show that it finds the training rows of an overfit model (AUC above 0.6) and stays close to guessing on a well-regularized one. The existing tests that use `DecisionTreeClassifier` and `DecisionTreeRegressor` show how.
4. **Rerun the benchmarks** with `python benchmarks/run_benchmarks.py` (about 2 minutes). Include the updated `benchmarks/results.md` and `benchmarks/results.json` in your pull request.

Some attacks, like LiRA, train extra copies of the model and don't fit this pattern. For those, describe your plan in the issue and we'll work out the design together.

## Check your work

GitHub runs these checks on every pull request, on Python 3.9 to 3.14. Run them yourself first:

```bash
pytest                          # runs the tests, takes a few seconds
ruff check .                    # finds common mistakes
ruff format --check .           # checks formatting; run `ruff format .` to fix it
trainleak demo --n-reference 4  # runs a full audit from start to finish
```

## Open a pull request

1. Work on a new branch from `main`, named after the change, such as `label-only-attack`.
2. Keep each pull request to one change. Small pull requests get reviewed faster.
3. Write commit messages as commands: `Add label-only attack`, not `Added label-only attack`.
4. Fill in the pull request template and link the issue it solves, for example `Closes #12`.

## Using AI tools

You may use AI tools to understand a problem, explore the code and help write a change. You are still the author, and you are responsible for everything you submit. We are strict about this:

- **Code.** Submit only code you understand and can explain line by line, and run it and check its results yourself.
- **Writing.** Write issues, pull request descriptions, review replies and comments, on GitHub and in the code, in your own words.
- **Review.** Answer questions about your change yourself.

Pull requests and issues that don't follow these rules will be closed.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind and assume good intent.
