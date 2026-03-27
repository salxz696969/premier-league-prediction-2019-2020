"""Helper utility functions."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from ..config.config import COUNTRY_CODES, TEAM_ALIASES


def log(msg: str) -> None:
    """Print an informational message."""
    print(f"[INFO] {msg}")


def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name (case-insensitive), or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_team(name) -> str:
    """Lowercase, strip punctuation, remove leading/trailing country codes, apply aliases."""
    s = re.sub(r"[^a-z0-9\s]", " ", str(name).strip().lower())
    tokens = [t for t in s.split() if t]
    if tokens and tokens[0] in COUNTRY_CODES:
        tokens = tokens[1:]
    if tokens and tokens[-1] in COUNTRY_CODES:
        tokens = tokens[:-1]
    s = " ".join(tokens)
    return TEAM_ALIASES.get(s, s)


def col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a column as a Series, or a NaN Series if missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)
