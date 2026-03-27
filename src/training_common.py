"""Shared training workflow utilities for all model variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.config.config import ROLLING_WINDOW
from src.features import (
    attach_context_features,
    attach_dynamic_motivation,
    attach_fatigue,
    attach_h2h_features,
    attach_position_features,
    attach_referee_features,
    attach_style_engineered_features,
    build_rolling_features,
)
from src.pipeline import stratified_group_split
from src.utils import (
    col_or_nan,
    load_competition_matches,
    load_position_map,
    load_premier_league,
    log,
    normalize_team,
    parse_home_away,
    pick_first_existing,
)


def load_engineered_dataset(data_root: Path) -> pd.DataFrame:
    """Load source CSV files and attach all engineered features."""
    additional = data_root / "additional_data"

    df = load_premier_league(data_root, additional)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Result"] = df["Result"].astype(str).str.lower().str.strip()
    df["Team"] = df["Team"].map(normalize_team)
    df["Opponent"] = df["Opponent"].map(normalize_team)
    df["xG"] = pd.to_numeric(df["xG"], errors="coerce")
    df["xGA"] = pd.to_numeric(df["xGA"], errors="coerce")

    # Temporary helper columns used by rolling builder
    df["_scored"] = pd.to_numeric(col_or_nan(df, "Scored"), errors="coerce")
    df["_conceded"] = pd.to_numeric(col_or_nan(df, "Conceded"), errors="coerce")
    df["_win_val"] = df["Result"].map({"w": 1.0, "d": 0.5, "l": 0.0})
    df["_ppda"] = pd.to_numeric(df.get("PPDA", pd.Series(dtype=float)), errors="coerce")
    if "xpts" in df.columns:
        df["_xpts"] = pd.to_numeric(df["xpts"], errors="coerce")

    df["home_advantage"] = (
        parse_home_away(df).replace({"home": "h", "away": "a"}).fillna("a")
    )

    df = df[
        df["Date"].notna()
        & df["Result"].isin(["w", "d", "l"])
        & df["Team"].notna()
        & df["Opponent"].notna()
    ].copy()
    log(f"Rows after filtering: {len(df):,}")

    comp_df = pd.concat(
        [
            load_competition_matches(data_root / "champion_league.csv", "ucl"),
            load_competition_matches(
                pick_first_existing(
                    data_root / "europa_league.csv", data_root / "europe_league.csv"
                ),
                "uel",
            ),
            load_competition_matches(data_root / "fa_cup.csv", "fa"),
            load_competition_matches(data_root / "carabao.csv", "carabao"),
        ],
        ignore_index=True,
    )
    df = attach_fatigue(df, comp_df)

    df = build_rolling_features(df)
    df.drop(
        columns=["_scored", "_conceded", "_win_val", "_ppda", "_xpts"],
        errors="ignore",
        inplace=True,
    )

    pos_map = load_position_map(data_root / "league_position_after20.csv")
    if not pos_map:
        log("Position features will be NaN (file missing or unreadable)")
    df = attach_position_features(df, pos_map)

    df = attach_h2h_features(df)
    df, _ = attach_context_features(df)
    df = attach_referee_features(df)
    df = attach_dynamic_motivation(df)
    df = attach_style_engineered_features(df)

    return df


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return numeric, categorical, and combined feature column lists."""
    w = ROLLING_WINDOW
    base_numeric_cols = ["xG", "xGA"]
    rolling_keep_cols = [
        f"rolling_xpts_{w}",
        f"rolling_ppda_{w}",
        f"rolling_scored_{w}",
        f"rolling_conceded_{w}",
        "rolling_xG_style_adj",
        "rolling_xGA_style_adj",
    ]
    context_keep_cols = [
        "Possession_Style_Weighted",
        "SCA_Style_Weighted",
        "Dribbles_Style_Weighted",
        "Final_Third_Style_Weighted",
        "Possession_Style_Delta",
        "Final_Third_Entries_Allowed",
        "Aerial_Battles_Won_Pct",
        "Save_Pct",
        "PPDA",
        "Allowed_PPDA",
    ]
    psych_cols = [
        "Referee_Bias_Score",
        "Motivation_Score",
        "referee_style_impact",
    ]
    interaction_cols = ["xG_diff", "form_vs_strength", "momentum"]

    numeric_cols = [
        c
        for c in (
            base_numeric_cols
            + ["h2h_win_rate"]
            + rolling_keep_cols
            + context_keep_cols
            + psych_cols
            + interaction_cols
        )
        if c in df.columns
    ]
    cat_cols = ["home_advantage"]
    all_feat_cols = numeric_cols + cat_cols

    log(
        f"Features: {len(all_feat_cols)} total  "
        f"(numeric={len(numeric_cols)}, categorical={len(cat_cols)})"
    )
    return numeric_cols, cat_cols, all_feat_cols


def split_features_and_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    val_frac: float = 0.20,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split dataset into train/val/test and return X/y for each split."""
    train_df, val_df, test_df = stratified_group_split(
        df, val_frac=val_frac, test_frac=test_frac
    )

    X_train, y_train = train_df[feature_cols], train_df["Result"]
    X_val, y_val = val_df[feature_cols], val_df["Result"]
    X_test, y_test = test_df[feature_cols], test_df["Result"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def print_model_report(
    model_name: str,
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    include_feature_importance: bool = False,
) -> None:
    """Print common evaluation report for all classifiers."""
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {model_name}  features={n_features}")
    print(sep)
    print(f"  Train : {n_train:>6,} rows")
    print(f"  Val   : {n_val:>6,} rows")
    print(f"  Test  : {n_test:>6,} rows")
    print(f"\n  Validation Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy       : {test_acc:.4f}")
    print(f"\nValidation Report:\n{classification_report(y_val, y_val_pred, digits=4)}")
    print(f"Test Report:\n{classification_report(y_test, y_test_pred, digits=4)}")

    if not include_feature_importance:
        return

    rf = model.named_steps.get("clf")
    preprocess = model.named_steps.get("preprocess")
    if rf is None or preprocess is None or not hasattr(rf, "feature_importances_"):
        return

    feat_names = [
        name.split("__", 1)[-1] for name in preprocess.get_feature_names_out()
    ]
    importance_df = (
        pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 15 feature importances:")
    print(importance_df.head(15).to_string(index=False))
