"""Basic feature engineering and attachment functions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..config.config import ROLLING_WINDOW, CONTEXT_COLS
from ..utils import col_or_nan, find_col, log


def attach_fatigue(df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative pre-match fatigue score via backward merge-asof."""
    if comp_df.empty:
        df["fatigue_score"] = 0.0
        log("No competition data — fatigue_score = 0")
        return df

    comp_df = (
        comp_df.dropna(subset=["team", "date"]).sort_values(["team", "date"]).copy()
    )
    comp_df["cum_weight"] = comp_df.groupby("team")["weight"].cumsum()

    left = (
        df.sort_values(["Team", "Date"])
        .reset_index()
        .rename(columns={"index": "_orig_idx"})
    )
    right = comp_df.rename(columns={"team": "Team"}).sort_values(["Team", "date"])

    left = pd.merge_asof(
        left,
        right[["Team", "date", "cum_weight"]],
        left_on="Date",
        right_on="date",
        by="Team",
        direction="backward",
        allow_exact_matches=False,
    )
    left["fatigue_score"] = left["cum_weight"].fillna(0.0)
    df["fatigue_score"] = left.sort_values("_orig_idx")["fatigue_score"].to_numpy()
    log(f"Fatigue scores attached from {len(comp_df):,} competition rows")
    return df


def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Shift-1 rolling mean — no target leakage."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def build_rolling_features(
    df: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Build rolling form features with no leakage."""
    df = df.sort_values(["Team", "Date"]).copy()
    grp = df.groupby("Team", group_keys=False)

    roll = lambda col: grp[col].transform(lambda s: _lagged_rolling_mean(s, window))

    df[f"rolling_xG_{window}"] = roll("xG")
    df[f"rolling_xGA_{window}"] = roll("xGA")
    df[f"rolling_scored_{window}"] = roll("_scored")
    df[f"rolling_conceded_{window}"] = roll("_conceded")
    df[f"rolling_win_rate_{window}"] = roll("_win_val")
    df[f"rolling_ppda_{window}"] = roll("_ppda")

    if "_xpts" in df.columns:
        df[f"rolling_xpts_{window}"] = roll("_xpts")

    log(f"Rolling features built (window={window})")
    return df


def attach_position_features(df: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    """Attach league position features for team and opponent."""
    if not pos_map:
        df["team_position"] = np.nan
        df["opponent_position"] = np.nan
        df["position_gap"] = np.nan
        return df
    df = df.copy()
    df["team_position"] = df["Team"].map(pos_map)
    df["opponent_position"] = df["Opponent"].map(pos_map)
    df["position_gap"] = df["opponent_position"] - df["team_position"]
    return df


def attach_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach head-to-head win rate features."""
    df = df.sort_values("Date").copy()
    df["_h2h_key"] = df[["Team", "Opponent"]].apply(
        lambda r: "|".join(sorted([r["Team"], r["Opponent"]])), axis=1
    )
    df["_h2h_win"] = (df["Result"] == "w").astype(float)
    df["h2h_win_rate"] = (
        df.groupby("_h2h_key")["_h2h_win"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.5)
    )
    df.drop(columns=["_h2h_key", "_h2h_win"], inplace=True)
    log("H2H win rate attached")
    return df


def attach_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach in-match context features (possession, SCA, dribbles, etc.)."""
    added = []
    for canonical, candidates in CONTEXT_COLS.items():
        src = find_col(df, *candidates)
        if src is not None:
            df[canonical] = pd.to_numeric(df[src], errors="coerce")
            added.append(canonical)
    if added:
        log(f"Context features: {added}")
    return df, added
