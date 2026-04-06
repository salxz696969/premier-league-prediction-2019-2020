# Project state — 6 April 2026

This note updates the project status after EDA and comparison work completed after `current_state_040426.md`.

## What changed since 4 April

- Added `notebooks/eda2.ipynb` with a simpler, narrative EDA flow focused on feature understanding.
- Clarified data perspective in EDA: `Team` is treated as home-side perspective and `Opponent` as away-side label.
- Added scenario-based model comparison output in `data/model_comparison_results.csv`.
- Added a dedicated visualization notebook `notebooks/model_comparison.ipynb` to compare model performance across experiment settings.
- Added architecture documentation in root `architecture.md`.

## Current models in use

All production-style scripts live in `src/models/` and share `src/training_common.py`:

- `RandomForestClassifier` (`n_estimators=200`, balanced classes)
- `DecisionTreeClassifier` (`max_depth=12`, `min_samples_leaf=5`, balanced classes)
- `LogisticRegression` (linear baseline, `max_iter=1000`, balanced classes)
- `XGBoostClassifier` (in notebook workflows)

## Current baseline performance (modeling notebook)

From `notebooks/modeling.ipynb` (current reference run):

| Model | Validation | Test |
|---|---:|---:|
| RandomForestClassifier | 0.5519 | 0.5974 |
| DecisionTreeClassifier | 0.5162 | 0.5422 |
| LogisticRegression | 0.5584 | 0.5714 |
| XGBoostClassifier | 0.5649 | 0.5649 |

## Scenario comparison now tracked

`notebooks/model_comparison.ipynb` compares:

- `Full` (additional data on, group-leakage-safe split, cross-validation on)
- `With leakage true`
- `Cross validation off` (aligned to modeling notebook baseline)
- `Additional data off`

The notebook includes bar charts and delta tables to make scenario effects easier to read than text output.

## EDA state

Current EDA is split by intent:

- `notebooks/eda.ipynb`: broader technical EDA and checks.
- `notebooks/eda2.ipynb`: simpler explanatory EDA for communicating feature behavior and assumptions.

## Data collection and cleaning status

Data cleaning in the current repo remains partly manual and partly AI-assisted.  
Automation scripts for end-to-end data collection/cleaning are in progress and intentionally left open for future implementation.

## Practical next steps

- Lock one canonical experiment runner so notebook and script metrics match exactly across scenarios.
- Add automated checks for leakage-safe split behavior.
- Add a reproducible data build step when collection/cleaning automation is ready.
