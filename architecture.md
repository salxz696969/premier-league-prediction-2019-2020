# Project Architecture Guide (Beginner-Friendly)

This document explains the project from zero knowledge to implementation details.
It is written to help you understand:

1. the core data-science concepts used here,
2. the exact step-by-step workflow in this codebase,
3. the models and why they are used,
4. the current results from notebooks.

---

## 1) Big picture: what this project is doing

Goal: predict Premier League match outcome labels:

- `w` = win
- `d` = draw
- `l` = loss

This is a **3-class classification** problem.  
Input = team-level match statistics and engineered context features.  
Output = predicted class (`w`, `d`, or `l`).

---

## 2) Core concepts and keywords (plain language)

### Classification
Predicting a category instead of a number. Here, category = `w/d/l`.

### Feature
A model input column. Example: `xG`, `xGA`, `PPDA`, fatigue score.

### Feature engineering
Creating new useful columns from raw data. Example: rolling form, H2H win rate, style-adjusted metrics.

### EDA (Exploratory Data Analysis)
Investigation before modeling:
- check data quality,
- find useful signals,
- detect risks (leakage, bias, bad assumptions).

### Leakage
Model gets information it would not have in real prediction time.
Leakage makes scores look better than they should.

### Train / Validation / Test
- **Train**: fit model parameters.
- **Validation**: tune/compare setups.
- **Test**: final unbiased check.

### Cross-validation (CV)
Repeat training on multiple folds to estimate stability, not just one split.

### Imputation
Filling missing values:
- numeric with median,
- categorical with most frequent value.

### One-hot encoding
Convert categorical values into numeric indicator columns.

### Class imbalance
Some classes appear less often (draws are usually harder); this affects metrics.

---

## 3) Repository map (what each part does)

- `data/`: all CSV sources (league + competitions + optional additional data).
- `src/training_common.py`: shared end-to-end training flow.
- `src/features/`: feature engineering modules.
- `src/pipeline/splitting.py`: leakage-safe split logic.
- `src/models/`: model scripts (Random Forest, Decision Tree, Logistic Regression).
- `notebooks/modeling.ipynb`: baseline modeling and reports.
- `notebooks/eda.ipynb`, `notebooks/eda2.ipynb`: EDA.
- `notebooks/model_comparison.ipynb`: visual scenario comparison.
- `data/model_comparison_results.csv`: scenario experiment table.

---

## 4) Data in this project

## 4.1 Main data files

Primary league table:
- `data/premier_league.csv`

Optional expanded league rows:
- `data/additional_data/premier_league.csv`

Competition data (fatigue context):
- `data/champion_league.csv`
- `data/europa_league.csv` (fallback name supported in code)
- `data/fa_cup.csv`
- `data/carabao.csv`

Auxiliary team ranking table:
- `data/league_position_after20.csv`

## 4.2 Data collection/cleaning status

Current process is partly manual and partly AI-assisted.  
Automation scripts for full collection + cleaning are still in progress (planned future work).

---

## 5) Step-by-step workflow (what we do and why)

This is the practical pipeline implemented in `src/training_common.py`.

## Step 1: Load and merge

Load main league CSV and optional additional league CSV, align shared columns, concatenate, deduplicate.

Why: use maximum available rows while keeping schema consistent.

## Step 2: Normalize core fields

Standardize:
- `Date` parsing,
- `Result` cleanup to `{w,d,l}`,
- team/opponent names normalization,
- numeric conversion (`xG`, `xGA`, etc.),
- derive `home_advantage` if available.

Why: prevent model errors from inconsistent text/types.

## Step 3: Filter trainable rows

Keep rows with valid date, valid label, valid team/opponent.

Why: avoid fitting on broken rows.

## Step 4: Add fatigue context

Load cup/Europe fixtures and attach fatigue score to league rows.

Why: schedule congestion can affect performance.

## Step 5: Build rolling form features

Compute rolling windows (default 5) for key form stats.

Why: recent form matters more than distant matches.

## Step 6: Add football-context features

Attach:
- league position features,
- H2H rate,
- possession/creation/defensive context.

Why: include tactical and matchup information.

## Step 7: Add advanced style/psychology signals

Attach:
- style-aware transformed context,
- referee interaction,
- dynamic motivation score.

Why: inject richer domain assumptions beyond raw box-score stats.

## Step 8: Select final feature set

Current design uses:
- ~25 numeric features (when all available),
- 1 categorical feature (`home_advantage`).

## Step 9: Split data (leakage-safe default)

Use grouped split so both rows of the same fixture stay in the same fold.

Why: avoid same-match leakage between train/val/test.

## Step 10: Train + evaluate models

Train pipelines with shared preprocessing and compare metrics.

---

## 6) EDA architecture and what it tells us

### `notebooks/eda.ipynb`
Broader technical EDA and sanity checks.

### `notebooks/eda2.ipynb`
Simpler narrative EDA focused on understanding:
- outcome distribution and trends,
- xG realism (`xG` vs goals, `xGA` vs conceded),
- fatigue buckets,
- weekly scoring trends,
- late-season pressure proxy.

How EDA helps this project:
- confirms important features are meaningful,
- identifies limitations (for example home/away perspective assumptions),
- informs model-comparison scenarios.

---

## 7) Model architecture (implemented models)

All models use the same preprocessing:

- numeric: `SimpleImputer(strategy="median")`
- categorical: mode imputer + one-hot encoding

Models:

1. `RandomForestClassifier`
   - 200 trees, class-balanced.
   - strong non-linear baseline.

2. `DecisionTreeClassifier`
   - constrained depth/leaf for interpretability and regularization.

3. `LogisticRegression`
   - linear baseline, class-balanced.
   - useful sanity reference.

4. `XGBoostClassifier` (notebook-driven)
   - gradient boosting alternative.
   - often competitive on tabular data.

Why this mix:
- one linear baseline,
- one simple tree,
- one ensemble tree,
- one boosted model.
This gives a good trade-off between interpretability and predictive power.

---

## 8) Baseline results (from `notebooks/modeling.ipynb`)

Reference run currently used by docs:

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| RandomForestClassifier | 0.5519 | 0.5974 |
| DecisionTreeClassifier | 0.5162 | 0.5422 |
| LogisticRegression | 0.5584 | 0.5714 |
| XGBoostClassifier | 0.5649 | 0.5649 |

Reading this correctly:
- performance is moderate (realistic for football outcomes),
- draws remain hard,
- no model dominates every metric.

---

## 9) Scenario comparison (from `model_comparison`)

Scenarios compared in `notebooks/model_comparison.ipynb`:

- `Cross validation off` (baseline alignment to `modeling.ipynb`)
- `Full` (CV enabled)
- `With leakage true`
- `Additional data off`

Main observations:

- `Full` adds CV visibility (stability view), not just single split scores.
- `With leakage true` changes behavior and is included to show why split strategy matters.
- `Additional data off` can produce higher scores in this run, but this should be interpreted carefully because:
  - dataset becomes much smaller,
  - split composition can become easier,
  - it does not automatically mean less data is better in general.

---

## 10) Why the implementation is done this way

- **Shared training flow** (`training_common`) keeps model scripts consistent.
- **Modular features** make it easier to evolve domain logic.
- **Grouped split** protects against a common leakage pattern in team-pair data.
- **Notebook + script combo** supports both exploration and reproducible runs.
- **Scenario notebook** makes model behavior easier to understand than raw logs.

---

## 11) Known limitations

- Data collection/cleaning is not fully automated yet.
- Some advanced features are heuristic and may need season-by-season recalibration.
- Group split is leakage-safe, but not a strict time-forward backtest.
- Home/away interpretation depends on source-table perspective assumptions.

---

## 12) Recommended next upgrades

1. Add one canonical experiment runner shared by scripts and notebooks.
2. Add tests for split integrity and feature expectations.
3. Finalize automated data collection/cleaning pipeline.
4. Add strict time-based backtesting mode.
5. Track additional metrics (macro-F1, class-wise recall for draws), not only accuracy.
