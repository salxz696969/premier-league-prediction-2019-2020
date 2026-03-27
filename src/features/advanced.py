"""Advanced feature engineering with style awareness and interaction terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.config import ROLLING_WINDOW, TEAM_BOOST_OVERRIDES
from ..utils import col_or_nan, log
from .style_analysis import (
    calculate_fixture_importance,
    calculate_opponent_strength,
    calculate_recent_form,
    classify_team_style,
    estimate_squad_depth,
)
from .style_calculations import (
    calculate_style_weighted_context,
    opponent_style_interaction,
    style_defense_expectation,
    style_xg_expectation,
)


def build_style_aware_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features with style-aware interpretation."""
    df = df.sort_values(["Team", "Date"]).copy()

    def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    w = ROLLING_WINDOW
    grp = df.groupby("Team", group_keys=False)
    roll = lambda col: grp[col].transform(lambda s: _lagged_rolling_mean(s, w))

    # Base rolling features
    df[f"rolling_xG_{w}"] = roll("xG")
    df[f"rolling_xGA_{w}"] = roll("xGA")
    df[f"rolling_scored_{w}"] = roll("_scored")
    df[f"rolling_conceded_{w}"] = roll("_conceded")
    df[f"rolling_win_rate_{w}"] = roll("_win_val")
    df[f"rolling_ppda_{w}"] = roll("_ppda")

    if "_xpts" in df.columns:
        df[f"rolling_xpts_{w}"] = roll("_xpts")

    log(f"Style-aware rolling features built (window={w})")
    return df


def attach_style_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject style-aware engineered features directly into training matrix inputs.
    All historical computations use Date < current match Date to avoid leakage.
    """
    out = df.sort_values(["Date", "Team", "Opponent"]).copy()
    w = ROLLING_WINDOW

    referee_style_impact = []
    xg_diff = []
    form_vs_strength = []
    momentum = []

    possession_style_weighted = []
    sca_style_weighted = []
    dribbles_style_weighted = []
    final_third_style_weighted = []
    possession_style_delta = []
    rolling_xg_style_adj = []
    rolling_xga_style_adj = []

    for _, row in out.iterrows():
        current_date = row["Date"]
        history = out[out["Date"] < current_date]

        team = row["Team"]
        opponent = row["Opponent"]

        team_style = classify_team_style(history, team, window=10)
        opp_style = classify_team_style(history, opponent, window=10)

        recent_form = calculate_recent_form(history, team, window=5)
        opp_strength = calculate_opponent_strength(history, opponent)

        _ = opponent_style_interaction(team_style, opp_style)

        weights = calculate_style_weighted_context(row, team_style)

        pos = pd.to_numeric(
            pd.Series([row.get("Possession", np.nan)]), errors="coerce"
        ).iloc[0]
        sca = pd.to_numeric(
            pd.Series([row.get("Shot_Creating_Actions", np.nan)]), errors="coerce"
        ).iloc[0]
        dri = pd.to_numeric(
            pd.Series([row.get("Successful_Dribbles", np.nan)]), errors="coerce"
        ).iloc[0]
        fte = pd.to_numeric(
            pd.Series([row.get("Final_Third_Entries", np.nan)]), errors="coerce"
        ).iloc[0]

        possession_style_weighted.append(
            pos * weights.get("Possession", 1.0) if pd.notna(pos) else np.nan
        )
        sca_style_weighted.append(
            sca * weights.get("Shot_Creating_Actions", 1.0) if pd.notna(sca) else np.nan
        )
        dribbles_style_weighted.append(
            dri * weights.get("Successful_Dribbles", 1.0) if pd.notna(dri) else np.nan
        )
        final_third_style_weighted.append(
            fte * weights.get("Final_Third_Entries", 1.0) if pd.notna(fte) else np.nan
        )
        possession_style_delta.append(
            pos - float(team_style.get("possession_pct", 50.0))
            if pd.notna(pos)
            else np.nan
        )

        rolling_xg = pd.to_numeric(
            pd.Series([row.get(f"rolling_xG_{w}", np.nan)]), errors="coerce"
        ).iloc[0]
        rolling_xga = pd.to_numeric(
            pd.Series([row.get(f"rolling_xGA_{w}", np.nan)]), errors="coerce"
        ).iloc[0]

        xg_exp = style_xg_expectation(team_style)
        xga_exp = style_defense_expectation(team_style)

        xg_adj_val = rolling_xg / max(xg_exp, 1e-6) if pd.notna(rolling_xg) else np.nan
        xga_adj_val = rolling_xga * xga_exp if pd.notna(rolling_xga) else np.nan
        rolling_xg_style_adj.append(xg_adj_val)
        rolling_xga_style_adj.append(xga_adj_val)

        xg_diff.append(
            (xg_adj_val - xga_adj_val)
            if pd.notna(xg_adj_val) and pd.notna(xga_adj_val)
            else np.nan
        )

        _depth = max(estimate_squad_depth(history, team, window=15), 0.10)

        home_flag = str(row.get("home_advantage", "a")).lower() == "h"
        team_fouls = (
            row.get("Home_Fouls", np.nan)
            if home_flag
            else row.get("Away_Fouls", np.nan)
        )
        team_fouls = (
            pd.to_numeric(pd.Series([team_fouls]), errors="coerce").fillna(0.0).iloc[0]
        )
        ppda_val = (
            pd.to_numeric(pd.Series([row.get("PPDA", np.nan)]), errors="coerce")
            .fillna(6.0)
            .iloc[0]
        )
        aggression = team_fouls + max(0.0, 8.0 - ppda_val)
        ref_bias = (
            pd.to_numeric(
                pd.Series([row.get("Referee_Bias_Score", 0.0)]), errors="coerce"
            )
            .fillna(0.0)
            .iloc[0]
        )
        referee_style_impact.append(ref_bias * (1.0 + aggression / 10.0))

        rp = pd.to_numeric(
            pd.Series([row.get(f"rolling_win_rate_{w}", np.nan)]), errors="coerce"
        ).iloc[0]
        rp = (rp * 3.0) if pd.notna(rp) else (recent_form["points_per_game"] * 3.0)

        form_vs_strength.append(recent_form["points_per_game"] * opp_strength)
        momentum.append(rp + recent_form["streak"])

    out["Possession_Style_Weighted"] = possession_style_weighted
    out["SCA_Style_Weighted"] = sca_style_weighted
    out["Dribbles_Style_Weighted"] = dribbles_style_weighted
    out["Final_Third_Style_Weighted"] = final_third_style_weighted
    out["Possession_Style_Delta"] = possession_style_delta

    out["rolling_xG_style_adj"] = rolling_xg_style_adj
    out["rolling_xGA_style_adj"] = rolling_xga_style_adj

    out["referee_style_impact"] = referee_style_impact

    out["xG_diff"] = xg_diff
    out["form_vs_strength"] = form_vs_strength
    out["momentum"] = momentum

    log("Style engineered features attached")
    return out


def attach_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add style-aware referee bias features."""
    if "Referee" not in df.columns:
        df["Referee_Bias_Score"] = 0.0
        log("No Referee column — Referee_Bias_Score = 0")
        return df

    out = df.sort_values(["Date", "Team", "Opponent"]).copy()
    out["_ref"] = (
        out["Referee"].astype(str).str.lower().str.strip().replace("", "unknown")
    )

    home_mask = out["home_advantage"].astype(str).str.lower().eq("h")

    home_y = pd.to_numeric(col_or_nan(out, "Home_Yellow"), errors="coerce")
    home_r = pd.to_numeric(col_or_nan(out, "Home_Red"), errors="coerce")
    away_y = pd.to_numeric(col_or_nan(out, "Away_Yellow"), errors="coerce")
    away_r = pd.to_numeric(col_or_nan(out, "Away_Red"), errors="coerce")

    # Weighted card pressure: red = 2 yellows
    home_cards = home_y.fillna(0.0) + 2.0 * home_r.fillna(0.0)
    away_cards = away_y.fillna(0.0) + 2.0 * away_r.fillna(0.0)

    team_cards = np.where(home_mask, home_cards, away_cards)
    opp_cards = np.where(home_mask, away_cards, home_cards)

    pk_for = pd.to_numeric(col_or_nan(out, "PK"), errors="coerce").fillna(0.0)
    pk_against = pd.to_numeric(col_or_nan(out, "PK_Allowed"), errors="coerce").fillna(
        0.0
    )

    # Raw signal: positive means favorable officiating
    raw_signal = (opp_cards - team_cards) + 2.0 * (pk_for - pk_against)
    raw_signal = pd.Series(raw_signal, index=out.index).fillna(0.0)

    # Normalize by referee
    std = raw_signal.std()
    out["_ref_signal_norm"] = (
        (raw_signal - raw_signal.mean()) / std if pd.notna(std) and std > 0 else 0.0
    )

    # Historical bias: per referee-team pair (shift-1 for no leakage)
    out["Referee_Bias_Score"] = (
        out.groupby(["_ref", "Team"])["_ref_signal_norm"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.0)
    )

    # Adjust for league situation: relegation-zone teams or title-chasers get boosted sensitivity
    relegation_mask = out["team_position"].fillna(10) >= 16
    title_mask = out["team_position"].fillna(10) <= 2
    situation_multiplier = np.where(
        relegation_mask | title_mask, 1.3, 1.0
    )  # More sensitive when stakes are high
    out["Referee_Bias_Score"] = out["Referee_Bias_Score"] * situation_multiplier

    out.drop(columns=["_ref", "_ref_signal_norm"], inplace=True)
    log("Style-aware referee bias features attached")
    return out


def attach_dynamic_motivation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamic motivation score with style awareness."""
    out = df.sort_values(["Date", "Team"]).copy()
    motivation_scores = []

    for _, row in out.iterrows():
        current_date = row["Date"]
        history = out[out["Date"] < current_date]

        team = row["Team"]
        opp = row["Opponent"]
        team_pos = row.get("team_position", np.nan)
        opp_pos = row.get("opponent_position", np.nan)

        # Classify styles from historical data
        team_style = classify_team_style(history, team, window=10)
        opp_style = classify_team_style(history, opp, window=10)

        # Recent form component (style-adjusted)
        form = calculate_recent_form(history, team, window=5)
        form_score = 0.3 * form["points_per_game"]
        form_score += 0.05 * max(form["streak"], -3)

        # League position context
        position_score = 0.0
        if pd.notna(team_pos):
            if team_pos == 1:
                position_score = 0.4
            elif team_pos <= 5:
                position_score = 0.25
            elif team_pos <= 8:
                position_score = 0.15
            elif team_pos >= 16:
                position_score = 0.5

        # Opponent strength (style matters)
        opp_strength = calculate_opponent_strength(history, opp)
        opp_score = 0.15 * opp_strength

        # Fixture importance
        fixture_score = 0.2 * calculate_fixture_importance(team_pos, opp_pos, row)

        # Style interaction: bonus/penalty
        style_interaction = opponent_style_interaction(team_style, opp_style)

        # Squad depth: reduces fatigue impact on motivation
        squad_depth = estimate_squad_depth(history, team, window=15)
        depth_factor = 0.9 + 0.2 * squad_depth  # [0.9, 1.1]

        # Team boosters
        boost = TEAM_BOOST_OVERRIDES.get(team.lower(), 0.0)

        # Combine components
        total_motivation = (
            form_score
            + position_score
            + opp_score
            + fixture_score
            + style_interaction
            + boost
        ) * depth_factor

        motivation_scores.append(min(1.0, max(0.0, total_motivation)))

    out["Motivation_Score"] = motivation_scores
    log("Dynamic style-aware motivation scores calculated")
    return out
