"""Data loading and I/O functions."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from .helpers import find_col, log, normalize_team
from ..config.config import COMP_WEIGHTS


def load_premier_league(base: Path, additional: Path) -> pd.DataFrame:
    """Load and concatenate primary + optional additional PL CSV."""
    dfs = [pd.read_csv(base / "premier_league.csv", low_memory=False)]

    extra = additional / "premier_league.csv"
    if extra.exists():
        try:
            dfs.append(pd.read_csv(extra, low_memory=False))
            log("Loaded additional_data/premier_league.csv")
        except Exception as exc:
            log(f"Could not load additional premier_league.csv: {exc}")

    common = list(set.intersection(*[set(d.columns) for d in dfs]))
    merged = (
        pd.concat([d[common] for d in dfs], ignore_index=True).drop_duplicates().copy()
    )
    log(f"Raw rows after concat + dedup: {len(merged):,}")
    return merged


def parse_home_away(df: pd.DataFrame) -> pd.Series:
    """Derive a normalised 'h'/'a' column from whatever side/venue column exists."""
    for col in ("h_a", "side"):
        if col in df.columns:
            return df[col].astype(str).str.lower().str.strip()
    if "Venue" in df.columns:
        return (
            df["Venue"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map({"home": "h", "away": "a"})
        )
    return pd.Series(["a"] * len(df), index=df.index)


def load_position_map(path: Path) -> dict[str, float]:
    """Return {normalised_team: league_position} from a CSV file."""
    if not path.exists():
        log(f"Position file not found: {path}")
        return {}

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    team_col = find_col(df, "team", "teamid", "club")
    pos_col = find_col(df, "position", "pos", "rank")

    if not team_col or not pos_col:
        raise ValueError(f"{path} must have Team and Position columns")

    df["team_norm"] = df[team_col].map(normalize_team)
    df["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    df = df.dropna(subset=["team_norm", "pos"])
    return dict(zip(df["team_norm"], df["pos"]))


def pick_first_existing(*paths: Path) -> Path:
    """Return the first path that exists, falling back to the first argument."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _infer_stage(comp: str, stage_text: str, dt: pd.Timestamp) -> str:
    """Infer competition stage from text and date."""
    s = str(stage_text).lower()
    if any(
        k in s for k in ["semi", "final", "quarter", "qf", "sf", "r16", "round of 16"]
    ):
        return (
            "late"
            if any(k in s for k in ["semi", "final", "quarter", "qf", "sf"])
            else "knockout"
        )
    if "group" in s:
        return "group"
    month = dt.month if pd.notna(dt) else 1
    if comp in ("ucl", "uel"):
        return (
            "group"
            if month in [9, 10, 11, 12]
            else ("late" if month == 8 else "knockout")
        )
    if comp in ("fa", "carabao"):
        return "late" if month >= 3 else "early"
    return "early"


def load_competition_matches(path: Path, comp: str) -> pd.DataFrame:
    """Parse a competition CSV into (team, date, weight) rows."""
    empty = pd.DataFrame(columns=["team", "date", "weight"])
    if not path.exists():
        log(f"Skip missing: {path.name}")
        return empty

    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    date_col = find_col(df, "date")
    home_col = find_col(df, "home team", "home_team", "home")
    away_col = find_col(df, "away team", "away_team", "away")
    stage_col = find_col(df, "stage", "round")

    if not all([date_col, home_col, away_col]):
        log(f"Skip malformed: {path.name}")
        return empty

    df["_date"] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
    df["_stage"] = df[stage_col] if stage_col else ""

    rows = []
    for _, row in df.iterrows():
        if pd.isna(row["_date"]):
            continue
        stage = _infer_stage(comp, row["_stage"], row["_date"])
        weight = COMP_WEIGHTS.get(comp, {}).get(stage, 0.0)
        for col in (home_col, away_col):
            if col is None:
                continue
            rows.append(
                {
                    "team": normalize_team(row[col]),
                    "date": row["_date"],
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows, columns=["team", "date", "weight"])
