# Premier League match outcome prediction

A data science project that predicts Premier League match outcomes (win, draw, loss) from team-perspective rows in historical match data. The pipeline loads league and cup fixtures, engineers form, tactical, fatigue, and psychological features, then trains scikit-learn and gradient-boosting classifiers with a leakage-aware train/validation/test split.

## Requirements

- Python 3.13 or newer
- [uv](https://github.com/astral-sh/uv) (recommended) or another environment manager that can install dependencies from `pyproject.toml`

## Setup

From the repository root:

```bash
uv sync
```

Or install the project in editable mode with your preferred tool using the dependencies listed in `pyproject.toml`.

## Data layout

Place CSV inputs under `data/`:

- `premier_league.csv` — primary league table (team-level rows per match)
- `additional_data/premier_league.csv` — optional extra league rows (merged and deduplicated with the primary file)
- `league_position_after20.csv` — league position features (optional; if missing, position-based signals degrade)
- Cup schedules for fatigue: `champion_league.csv`, `europa_league.csv` (or `europe_league.csv`), `fa_cup.csv`, `carabao.csv`

The feature pipeline expects columns such as `Date`, `Team`, `Opponent`, `Result`, `xG`, `xGA`, and optional advanced stats (possession, PPDA, referee-related fields, etc.) when present in the source files.

## Running models

Model entry points live under `src/models/` and share one dataset builder (`src/training_common.py`):

```bash
uv run python src/models/random_forest.py
uv run python src/models/linear_regression.py
uv run python src/models/decision_tree.py
```

Each script loads `data/`, applies the full feature stack, fits the model, and prints validation and test accuracy plus classification reports.

## Notebooks

`notebooks/v2.ipynb` is the current main modeling notebook and compares multiple classifiers (Random Forest, Decision Tree, logistic regression, XGBoost) with time-aware evaluation flow.

## Project layout

| Path | Role |
|------|------|
| `src/config/` | Constants (rolling window, team aliases, cup fatigue weights, context column aliases) |
| `src/features/` | Feature engineering: rolling form, H2H, context, style-aware terms, referee and motivation features |
| `src/pipeline/` | `stratified_group_split` — splits by match so home/away rows stay in the same fold |
| `src/utils/` | Loading Premier League and cup CSVs, team name normalization, helpers |
| `src/training_common.py` | Shared load/engineer/split/evaluate workflow |
| `src/models/` | Per-model training scripts |

## License and data

Match statistics and competition data are subject to the terms of their original sources. Use this repository for research and education in line with those terms.

## Further detail

See `current_state_040426.md` for an architecture snapshot, what is implemented today, and known limitations as of that note.
