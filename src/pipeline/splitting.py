"""Data splitting with leakage-safe group stratification."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def _validate_split_fractions(val_frac: float, test_frac: float) -> None:
    """Validate split fractions and raise helpful errors for invalid inputs."""
    if val_frac <= 0 or test_frac <= 0:
        raise ValueError("val_frac and test_frac must both be > 0")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")


def make_match_group_id(df: pd.DataFrame) -> pd.Series:
    """Create a unique ID for each match (both home and away rows)."""
    t1 = df["Team"].str.lower().str.strip()
    t2 = df["Opponent"].str.lower().str.strip()
    lo = t1.where(t1 <= t2, t2)
    hi = t2.where(t1 <= t2, t1)
    date_str = (
        pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("NA")
    )
    return date_str + "|" + lo + "|" + hi


def _group_strat_label(grp: pd.DataFrame) -> str:
    """Determine stratification label for a match group."""
    vc = grp["Result"].value_counts()
    if vc.empty:
        return "d"
    if "d" in vc.index and vc["d"] >= vc.max():
        return "d"
    return str(vc.idxmax())


def stratified_group_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split rows into train/validation/test using match-group stratification.

    Why group-based split:
    each real-world match appears as two rows (one per team). Grouping keeps
    both rows in the same split to avoid information leakage.
    """
    _validate_split_fractions(val_frac, test_frac)

    df = df.copy()
    df["_match_gid"] = make_match_group_id(df)

    grp_labels = (
        df.groupby("_match_gid")
        .apply(_group_strat_label)
        .reset_index()
        .rename(columns={0: "strat"})
    )

    g_train, g_temp = train_test_split(
        grp_labels,
        test_size=val_frac + test_frac,
        random_state=random_state,
        stratify=grp_labels["strat"],
    )
    # Split temp into validation and test while preserving requested ratio.
    relative_test = test_frac / (val_frac + test_frac)
    g_val, g_test = train_test_split(
        g_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=g_temp["strat"],
    )

    def _select(match_group_ids: pd.Series) -> pd.DataFrame:
        """Return shuffled rows for the provided set of match-group IDs."""
        return (
            df[df["_match_gid"].isin(set(match_group_ids))]
            .drop(columns=["_match_gid"])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    return (
        _select(g_train["_match_gid"]),
        _select(g_val["_match_gid"]),
        _select(g_test["_match_gid"]),
    )
