# Technical Modeling Specification (`v2.ipynb`)

This document is a technical specification of the current modeling system implemented in `notebooks/v2.ipynb`.

Scope:
- end-to-end data-to-metrics execution path,
- leakage controls and temporal validation design,
- feature-engineering internals,
- model construction and evaluation pathways,
- interpretation of the two reported benchmark regimes.

This is intentionally implementation-centric and aligned to the notebook code structure.

---

## 1. Execution topology

`v2.ipynb` is organized into 12 functional blocks:

1. Imports and global config.
2. Utility helpers.
3. Data loading and fixture ingestion.
4. Base normalization / row filtering.
5. Chronological split.
6. Lagged and weighted rolling feature builders.
7. Leakage-safe feature attachment functions.
8. Split-aware orchestrator.
9. Feature selection + leakage assertion.
10. Matrix assembly.
11. Preprocessors + model builders.
12. Evaluation (optional walk-forward CV + final held-out + current-match benchmark).

Unlike simple notebooks that engineer features once globally, `v2` explicitly rebuilds feature tables per split/fold to preserve causal ordering.

---

## 2. Global configuration and reproducibility controls

Key global toggles/parameters in `v2`:

- `USE_ADDITIONAL_DATA`: controls whether `data/additional_data/` is merged.
- `RUN_CV`: toggles walk-forward CV execution.
- `N_CV_SPLITS`: count of temporal folds for CV.
- `CV_N_ESTIMATORS`, `FINAL_N_ESTIMATORS`: estimator counts for boosting/forest.
- implicit deterministic seeds (`random_state=42`) across split/model builders.

Operational note:
- `RUN_CV=False` by default to avoid runtime explosion.
- held-out test evaluation always runs.

---

## 3. Data IO layer and schema harmonization

## 3.1 Utility primitives

Core notebook-local primitives:

- `log(msg)`: lightweight execution tracing.
- `find_col(df, *candidates)`: resilient column name resolver.
- `normalize_team(name)`: normalization of team strings (token cleanup, aliases).
- `col_or_nan(df, col)`: safe column retrieval fallback.

Design rationale:
- keep ingestion robust to minor schema drift and naming inconsistencies.

## 3.2 League ingestion

`load_premier_league(base)`:
- loads base league file,
- optionally merges additional league copy,
- intersects common columns before concat,
- deduplicates rows.

This prevents accidental widening by unmatched schemas.

## 3.3 Competition ingestion

`load_competition_matches(path, comp)`:
- parses date/team/stage fields from cup/europe files,
- infers stage via `_infer_stage`,
- emits event rows used by fatigue join logic.

`_load_competition_events(data_root)` unions UCL/UEL/FA/Carabao events.

---

## 4. Base table normalization pipeline

Functions:
- `_normalize_base_columns`
- `_add_rolling_helper_columns`
- `_filter_trainable_rows`
- `_drop_rolling_helper_columns`

Transform details:
- `Date` coerced to datetime.
- target normalized to canonical `w/d/l`.
- `Team` and `Opponent` normalized.
- major numeric columns coerced (non-numeric -> NaN).
- `home_advantage` derived from available source indicator columns.

Trainability filter enforces:
- valid date,
- valid target class,
- non-null team/opponent identifiers.

Why:
- split logic and rolling transforms require clean temporal/index keys.

---

## 5. Temporal split strategy

`time_based_split(df, val_frac=0.20, test_frac=0.20)`:

- sorts by date,
- partitions by unique date blocks,
- creates disjoint chronological train/val/test segments,
- raises on empty splits / boundary violations.

Important:
- this is stricter than random stratified splitting.
- chronology is preserved; no shuffle across time boundaries.

`stratified_group_split` compatibility path is routed to chronological behavior in `v2` context.

---

## 6. Rolling and weighted-lag transforms

Core builders:
- `_lagged_rolling_mean`
- `_lagged_weighted_mean`
- `build_rolling_features`

Implementation characteristics:
- `shift(1)` is applied before rollups (critical anti-leakage measure),
- supports multiple windows and weighted recency aggregation,
- creates form and momentum proxies (e.g., rolling goal-diff variants).

The weighted aggregator uses custom rolling `apply` to emphasize recent observations.

---

## 7. Leakage-safe feature attachment stack

`v2` attaches context in composable functions:

1. `attach_fatigue`
2. `attach_opponent_rolling_features`
3. `attach_position_features`
4. `attach_context_features`
5. `attach_h2h_features`
6. `attach_match_context_features`
7. `attach_dynamic_motivation`

### 7.1 Fatigue implementation notes

`attach_fatigue` computes event-weight accumulation with window decomposition:
- 7-day,
- 14-day,
- 30-day,
- derived recent-share,
- weighted aggregate fatigue score.

Window math uses cumulative-weight subtraction logic for efficiency.

### 7.2 Opponent rolling propagation

`attach_opponent_rolling_features` self-joins rolling columns against opponent/date keys and prefixes them (`opp_*`).

### 7.3 H2H with explicit history separation

`attach_h2h_features(df, history_df)`:
- allows historical context to be externalized,
- avoids using same-split future H2H states.

### 7.4 Advanced context path

`attach_match_context_features` builds row-level interaction features from:
- recent-form metrics,
- opponent strength,
- fixture importance,
- referee-pressure interaction.

`attach_dynamic_motivation` adds bounded motivation score using form-position-fixture components.

---

## 8. Split-aware orchestration (`engineer_split_features`)

This is the central anti-leakage mechanism.

`engineer_split_features(current_split_df, history_df, comp_df, pos_map)` executes:

1. helper columns on current split,
2. optional concatenation with history rows,
3. ordered application of all feature attachments on the combined frame,
4. strict re-selection of current split rows by index provenance.

Implementation techniques:
- index-preservation wrappers to avoid accidental row dropping/reordering,
- explicit `_is_split` markers for traceability,
- post-condition assertions for row-count integrity.

Practical effect:
- validation/test rows can use prior history features without seeing future rows.

---

## 9. Feature-space definition and leakage guardrails

### 9.1 Feature selection

`select_feature_columns(df)` constructs:
- base numeric block (`xG`, `xGA`, etc.),
- rolling/weighted engineered blocks,
- context/psychological blocks,
- categorical block (`home_advantage`),
- union into training matrix feature list.

Column inclusion is conditional on existence, enabling schema-resilient runs.

### 9.2 Leakage assertion

`assert_no_perfect_target_leakage(X, y)` checks for near-perfect deterministic proxies to target.

This is a fail-fast guard against accidental post-match columns leaking into training matrices.

---

## 10. Matrix assembly protocol

`v2` constructs matrices in two regimes:

1. leakage-safe feature regime (main reported benchmark),
2. expanded current-match feature regime (separate benchmark, intentionally labeled as potentially optimistic).

For missing cross-split columns, `NaN` columns are materialized to keep train/test feature spaces aligned.

---

## 11. Preprocessing architecture

Two preprocessor templates:

- `_base_preprocessor(num_feats, cat_feats)`:
  - numeric median imputation,
  - categorical most-frequent imputation + one-hot encoding.

- `_scaled_preprocessor(num_feats, cat_feats)`:
  - adds numeric scaling for scale-sensitive estimators (SVM/MLP).

This split avoids unnecessary scaling for tree ensembles while retaining correct conditioning for margin/NN models.

---

## 12. Model registry and builders

Builder functions:

- `build_logistic_regression_pipeline`
- `build_decision_tree_pipeline`
- `build_random_forest_pipeline`
- `build_svm_pipeline`
- `build_mlp_pipeline`
- `build_xgboost_pipeline`

All return sklearn-compatible pipelines with consistent feature preprocessing front-end.

Model registry in section 12:

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
SVM
MLPClassifier
XGBoostClassifier
```

XGBoost path is handled explicitly for label encoding constraints.

---

## 13. Evaluation subsystem

## 13.1 Optional walk-forward CV

`make_walk_forward_date_folds`:
- uses unique chronological dates,
- `TimeSeriesSplit` over date index,
- materializes fold-specific raw train/val tables.

`evaluate_model_on_fold`:
- re-engineers features per fold (critical),
- trains model on fold train,
- evaluates on fold val,
- logs:
  - accuracy,
  - class-wise F1 (`w/d/l`),
  - log-loss where supported.

Fold-level XGBoost path:
- label encoder fit inside each fold only,
- avoids cross-fold encoding leakage.

## 13.2 Final held-out test

`evaluate_models_on_test_set`:
- retrains each model on full train+val,
- evaluates once on held-out test,
- outputs sorted `final_test_accuracy`.

This is the primary reported benchmark.

---

## 14. Dual benchmark outputs in `v2`

`v2` emits two final accuracy tables:

1. `final_results_df`  
   Leakage-safe feature set benchmark.

2. `current_match_results_df`  
   Expanded current-match feature benchmark via `build_current_match_feature_columns`.

The second benchmark is explicitly marked as separate/possibly optimistic.

### 14.1 Current-match benchmark safeguards

`build_current_match_feature_columns`:
- starts from base feature set,
- discovers extra columns present in train/test,
- excludes helper/target columns,
- applies explicit goal/xG-style blocklist to avoid obvious post-match leakage fields,
- constructs numeric/categorical additions by dtype.

Despite blocklisting, this regime should be interpreted as a stress-test benchmark, not the canonical leakage-safe score.

---

## 15. Reported `v2` result snapshots (from notebook output)

Leakage-safe final held-out table (section output):
- RandomForestClassifier: ~0.4748
- XGBoostClassifier: ~0.4564
- LogisticRegression: ~0.4060
- DecisionTreeClassifier: ~0.3326

Current-match benchmark table (separate output block):
- XGBoostClassifier: ~0.7225
- RandomForestClassifier: ~0.6628
- DecisionTreeClassifier: ~0.6560
- LogisticRegression: ~0.6353

Interpretation constraint:
- do not compare these two regimes as if equivalent;
- only the leakage-safe final held-out block is canonical for realistic generalization.

---

## 16. Engineering patterns worth noting

`v2` uses several robust coding techniques:

- split-aware feature recomputation (instead of global precompute),
- index-preserving wrappers to avoid silent row drift,
- explicit anti-leakage assertions,
- model-specific preprocessing dispatch,
- model-specific label encoding lifecycle control (XGBoost),
- temporal fold generation on unique dates rather than row index.

These patterns materially improve methodological rigor over typical notebook pipelines.

---

## 17. Known technical gaps

- Data collection/cleaning automation scripts are placeholders and not yet integrated.
- Feature-generation runtime can be high due to repeated fold engineering.
- Additional calibration/threshold analysis is not yet part of default reporting.
- Current-match benchmark still requires strict interpretation discipline.

---

## 18. Recommended hardening roadmap

1. Extract `v2` logic into importable package modules (reduce notebook monolith coupling).
2. Add automated regression tests for:
   - split chronology,
   - no-leakage assertions,
   - feature-column stability.
3. Persist run metadata (seed/config/hash) with outputs for reproducible experiment lineage.
4. Add macro-F1 and draw-class recall as first-class ranking metrics, not accuracy-only ranking.
5. Introduce strict production profile that disallows current-match benchmark execution by default.

---

## 19. Operational summary

`v2.ipynb` implements a temporally-aware, split-safe, multi-model tabular pipeline with explicit leakage controls and dual benchmark outputs.
It is the authoritative modeling implementation for the current project state.
# Modeling Guide (Deep Dive on `v1.ipynb`)

This document explains the modeling system in detail, centered on `notebooks/v1.ipynb`.
It is intentionally more detailed than `architecture.md`, and is written to help you understand both:

- the concepts behind the pipeline, and
- exactly what the code is doing step by step.

---

## 1) What this project is trying to predict

The project predicts football match outcomes as a 3-class label:

- `w` (win),
- `d` (draw),
- `l` (loss).

The prediction target comes from the `Result` column in the league dataset.

This is a classic **multiclass classification** problem on tabular data.

---

## 2) Why `v1.ipynb` matters

`v1.ipynb` is not just a quick training notebook. It contains a full modeling framework:

- data loading,
- leak-aware feature engineering,
- chronological split logic,
- multiple model builders,
- fold-based evaluation,
- final held-out test evaluation.

Even when scores change between runs, the structure and logic in `v1` are the key reference for how modeling is designed.

---

## 3) Core concepts used in `v1` (plain-language glossary)

### A) Feature engineering
Creating better model inputs from raw columns.  
Examples in this project: rolling form, fatigue windows, H2H rates, motivation score.

### B) Leakage
Using information from future matches (directly or indirectly) when building features.
This produces unrealistically high performance.

### C) Time-aware split
`v1` uses chronological splitting so training rows come earlier than validation/test rows.
This better simulates real forecasting.

### D) Group awareness
Football data can have paired rows per fixture; splitting must avoid same-match information crossing train/test.

### E) Shared preprocessing + multiple classifiers
All models use similar preprocessing (imputation + encoding), then different classifiers are swapped in.
This makes model comparison fair.

---

## 4) Data inputs used by `v1`

Main and optional league data:

- `data/premier_league.csv`
- optional additional copy in `data/additional_data/premier_league.csv`

Competition data for fatigue:

- `champion_league.csv`
- `europa_league.csv`
- `fa_cup.csv`
- `carabao.csv`

Auxiliary:

- league position map (`league_position_after20.csv`)

---

## 5) Data cleaning and normalization in `v1`

`v1` includes notebook-level helpers for:

- logging and column lookup,
- team-name normalization,
- robust fallback when columns are missing.

Then core fields are normalized:

- `Date` -> datetime,
- `Result` -> clean string labels,
- numeric stats (`xG`, `xGA`, etc.) -> numeric dtype,
- home/away interpretation from available columns.

This stage is critical because noisy strings and mixed types break rolling features and model pipelines.

---

## 6) Feature engineering in `v1` (what is built and why)

`v1` has rich feature logic. The major blocks are below.

## 6.1 Fatigue features (competition workload)

Cup and European fixtures are converted into weighted events and then aggregated into windows:

- `fatigue_7d`
- `fatigue_14d`
- `fatigue_30d`
- `fatigue_recent_share`
- `fatigue_score` (weighted combination of windows)

Why: fixture congestion can hurt recovery, intensity, and consistency.

## 6.2 Rolling form features

Lagged rolling means are built to ensure only prior matches are used:

- rolling goals scored/conceded,
- rolling xG/xGA/xPts-like stats,
- rolling PPDA-style context.

`v1` also supports weighted recency variants.

Why: recent form is usually more predictive than long-term average.

## 6.3 Opponent rolling features

For each row, opponent rolling values are attached (prefixed as opponent features).

Why: match result depends on both team and opponent form.

## 6.4 Position features

League position map is attached for team and opponent.

Why: table strength/context provides useful prior information.

## 6.5 H2H features

Head-to-head rolling statistics are built:

- win rate,
- goal difference trend,
- points per game,
- match counts.

Why: some team matchups are consistently asymmetric.

## 6.6 Context features

Columns like possession, shot creation, dribbles, final-third entries, and defensive pressure are added when present.

Why: these capture style and match-control dimensions not seen in plain results.

## 6.7 Advanced context and psychology

`v1` adds:

- match-context engineered features (momentum, edge signals),
- referee bias/pressure proxy,
- dynamic motivation score (form + position + fixture context).

Why: include non-trivial football context that can shift outcomes around baseline strength.

---

## 7) Split strategy in `v1`

One of the most important design points in `v1`:

- time-based split is used (chronological boundaries),
- validation/test fractions are checked for validity,
- boundary overlap checks are enforced,
- utility aliases remain for backward compatibility.

This is intentional: it reduces leakage risk compared with random row splits.

---

## 8) Leak-safe fold engineering design

`v1` does something advanced: feature engineering can be done per split/fold while preserving historical context.

High-level idea:

1. Build features for current fold rows.
2. Optionally include prior history rows to compute lagged/fatigue context.
3. Keep only rows belonging to the current split after feature creation.

Why this matters:

- rolling/H2H/fatigue features are sensitive to history,
- naive global engineering can accidentally leak future context,
- split-aware engineering is safer and more realistic.

---

## 9) Models implemented in `v1`

`v1` compares four model families under a common framework:

1. **Logistic Regression**  
   Linear baseline. Good for sanity check and interpretability.

2. **Decision Tree**  
   Non-linear but simpler structure; can overfit if unconstrained.

3. **Random Forest**  
   Ensemble of trees, usually stronger and more stable than a single tree.

4. **XGBoost**  
   Boosted trees, often strong on tabular data with tuned setup.

All models use standardized preprocessing and are evaluated under the same scenario logic.

---

## 10) Evaluation workflow in `v1`

`v1` has two levels of evaluation:

## 10.1 Fold-level evaluation

Per fold, the notebook computes metrics (accuracy and class-wise summaries, plus log-loss where available).

## 10.2 Final held-out test evaluation

After model selection/assessment, each model is retrained on train+val and tested on held-out test.

`v1` output includes final test-accuracy tables.  
Because the notebook includes multiple runs/contexts, you may see more than one result block in outputs.

---

