# Project state — 4 April 2026

This note describes the architecture, intent, current implementation, and observed behaviour of the Premier League prediction codebase as of the date above. It is a snapshot for collaborators and future you, not a specification.

## Purpose

The goal is to predict the per-team match result label `Result` in `{w, d, l}` using only information that can be justified for a pre-match or team-centric view of a fixture. The problem is framed as three-class classification. The motivating use case is exploratory: quantify how much signal lives in engineered football features versus a naive baseline, and compare model families under the same preprocessing.

## High-level architecture

Data flows in one direction:

1. **Ingest** — `load_premier_league` merges the main `data/premier_league.csv` with optional `data/additional_data/premier_league.csv`, intersects columns, concatenates, and drops duplicates. Cup files are loaded separately for fatigue only.

2. **Normalize** — `training_common._normalize_base_columns` parses dates, normalises team names via `config.TEAM_ALIASES`, coerces `xG` / `xGA`, and derives `home_advantage` from `h_a`, `side`, or `Venue`.

3. **Filter** — Rows without a valid date, team, opponent, or result in `{w,d,l}` are removed.

4. **Fatigue** — `attach_fatigue` merges cup fixture timelines (Champions League, Europa League, FA Cup, Carabao) with weights from `config.COMP_WEIGHTS` and stage heuristics in `load_competition_matches`, so dense schedules increase a fatigue score before the league row’s date.

5. **Core features** — `build_rolling_features` uses a shift-aware rolling window (`ROLLING_WINDOW`, default 5) for points, goals, PPDA, and xG-style aggregates. Helpers `_scored`, `_conceded`, etc. are dropped after this step. `attach_position_features` maps table position from `league_position_after20.csv`. `attach_h2h_features` adds historical head-to-head win rate. `attach_context_features` pulls in possession, shot-creating actions, dribbles, final-third entries, defensive pressure, and goalkeeping proxies when columns exist (aliases in `CONTEXT_COLS`).

6. **Style and psychology** — `attach_referee_features`, `attach_dynamic_motivation`, and `attach_style_engineered_features` (in `advanced.py` with `style_analysis` and `style_calculations`) add style classification, weighted context, opponent interactions, referee bias-style scores, and a composite motivation score. Some parameters are domain heuristics (for example `TEAM_BOOST_OVERRIDES` for manual season-specific tweaks).

7. **Feature selection** — `select_feature_columns` defines a fixed list of numeric columns (roughly mid-twenties features when all columns are present) plus categorical `home_advantage`. Missing columns are skipped so the pipeline stays runnable on thinner CSVs.

8. **Split** — `stratified_group_split` builds a match group id from date and sorted team pair so both team rows from the same fixture fall into the same split. Groups are stratified by a coarse outcome label to keep class balance roughly stable across train, validation, and test. Default fractions are 70% train, 20% validation, 10% test.

9. **Model** — Each script in `src/models/` builds a `sklearn.pipeline.Pipeline`: median imputation for numerics, most-frequent imputation plus one-hot encoding for categorics, then the classifier (Random Forest with `class_weight='balanced'`, logistic regression, decision tree, etc.). The notebook wires the same preprocessor to additional estimators such as XGBoost.

## What is implemented now

- End-to-end training scripts that share `load_engineered_dataset`, `select_feature_columns`, `split_features_and_target`, and `print_model_report`.
- Rich feature layer: rolling form, cup fatigue, table position, H2H, multi-source context stats, style-weighted rolling xG/xGA adjustments, referee and motivation signals.
- Group-based splitting to reduce leakage from duplicate rows per match.
- exploratory notebook `notebooks/modeling.ipynb` for multi-model comparison and reporting.

What is not implemented or only lightly covered: automated tests, CLI flags, configuration via environment variables, logging to files, hyperparameter search, calibration of predicted probabilities, and strict temporal backtesting (splits are group-stratified but not necessarily rolling by season).

## Results (from the checked-in notebook run)

The following figures come from executed output embedded in `notebooks/modeling.ipynb` (same feature pipeline, group split, and comparable preprocessing). Slight differences will appear if you change data files, seeds, or library versions.

| Model | Validation accuracy | Test accuracy |
|--------|---------------------|---------------|
| RandomForest (200 trees) | 0.5519 | 0.5974 |
| DecisionTree | 0.5162 | 0.5422 |
| LogisticRegression | 0.5584 | 0.5714 |
| XGBoostClassifier | 0.5649 | 0.5649 |

Random Forest’s test accuracy is the highest in that run, but validation accuracy is lower than test, which is consistent with noise on a modest test size (308 rows in the test fold for that run) rather than a guarantee of generalisation. Treat leaderboard ordering as indicative, not definitive.

Feature importance from the same Random Forest run places `xGA`, `xG`, and `h2h_win_rate` among the strongest contributors, with rolling and style-adjusted terms and motivation also in the top tier.

## Flaws and limitations

- **Draw class** — Draws are the minority and hardest class; precision and recall for `d` stay weak compared to `w` and `l`, which is typical for football outcome models.
- **Signal ceiling** — Mid-to-high fifties or low sixties accuracy on held-out data is plausible for this task but still far from deterministic; many factors (injuries, lineups, luck) are not in the CSVs.
- **Heuristic and manual parts** — Fatigue weights, motivation, referee features, and `TEAM_BOOST_OVERRIDES` encode assumptions that may not transfer to other seasons or leagues without review.
- **Data dependency** — Missing optional columns silently drops features; comparing runs across machines requires the same `data/` contents.
- **Split design** — Group stratification removes same-match leakage but does not enforce a pure chronological cutoff; if the goal were strictly “predict future rounds from past rounds,” a time-based split would be stricter.

## Relation to older notes

`REFACTORING_SUMMARY.md` describes an earlier refactor from a monolithic script into modules and cites different accuracy numbers from an older run. Prefer this document and fresh notebook output for the current stack under `src/`.

## Suggested next steps (non-binding)

- Add pytest coverage for splitting, normalization, and one golden-path feature row.
- Expose data paths and split fractions via config or CLI.
- Replace ad-hoc prints with structured logging for reproducible experiment logs.
- Optional: time-based validation for reporting “true” forecasting performance.
