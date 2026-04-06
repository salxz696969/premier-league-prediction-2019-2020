# EDA current state — 6 April 2026

## Why EDA exists in this project

EDA here is used to validate whether football-domain features are meaningful before model tuning:

- confirm data quality and target balance,
- verify that expected-goals metrics carry real signal,
- inspect schedule congestion/fatigue effects,
- identify limitations that can bias interpretation.

## What `notebooks/eda2.ipynb` currently does

- Loads league and cup/Europe data from root `data/`.
- Standardizes key fields (`Date`, `Result`, core numeric metrics).
- Shows win/draw/loss mix and monthly trend.
- Analyzes home-side outcomes (with explicit note that rows are home-team perspective).
- Plots `xG` vs `Scored` and `xGA` vs `Conceded`.
- Tracks weekly goal trends with moving average.
- Builds a simple fatigue bucket (`0`, `1`, `2+` other-competition matches in prior 7 days).
- Compares outcomes/metrics across fatigue buckets.
- Adds a late-season pressure proxy (final 25% of season rows).
- Ends with actionable takeaways for modeling.

## Known EDA limitations (explicitly accepted)

- Home-vs-away causal comparison is limited by table perspective (home-side rows only).
- Fatigue signal is heuristic and depends on date/team-name normalization quality.
- EDA findings are associative, not causal.

## How EDA feeds modeling

- Prioritizes core features (`xG`, `xGA`, `Scored`, `Conceded`, fatigue bucket, phase proxy).
- Flags where split design and leakage prevention are critical.
- Informs which scenario comparisons to track in `model_comparison.ipynb`.

## Next EDA improvements

- Add uncertainty bands / confidence intervals for scenario claims.
- Add fixture-level merge for true home-vs-away tests when reliable join keys are finalized.
- Add automated data-quality checks once data pipeline automation is complete.
