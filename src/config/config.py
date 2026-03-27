"""Configuration and constants for the Premier League predictor."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature Engineering Parameters
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 5

# ---------------------------------------------------------------------------
# Team Normalization
# ---------------------------------------------------------------------------

COUNTRY_CODES = {
    "eng",
    "es",
    "de",
    "fr",
    "it",
    "nl",
    "pt",
    "ru",
    "ua",
    "gr",
    "be",
    "at",
    "hr",
    "rs",
    "tr",
    "ch",
    "dk",
    "no",
    "se",
    "pl",
    "cz",
    "sk",
}

TEAM_ALIASES: dict[str, str] = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "west ham": "west ham united",
    "newcastle": "newcastle united",
    "brighton": "brighton and hove albion",
}

# ---------------------------------------------------------------------------
# Competition Fatigue Weights
# ---------------------------------------------------------------------------

# Competition fatigue weights per stage
COMP_WEIGHTS: dict[str, dict[str, float]] = {
    "ucl": {"group": 1.0, "knockout": 2.0, "late": 3.0},
    "uel": {"group": 0.8, "knockout": 1.5, "late": 2.0},
    "fa": {"early": 0.5, "late": 1.5},
    "carabao": {"early": 0.3, "late": 1.0},
}

# ---------------------------------------------------------------------------
# Team-Specific Overrides
# ---------------------------------------------------------------------------

TEAM_BOOST_OVERRIDES: dict[str, float] = {
    # Season-specific manual adjustment requested by user.
    "liverpool": 0.40,
    "manchester city": 0.00,
}

# ---------------------------------------------------------------------------
# Context Feature Column Mappings
# ---------------------------------------------------------------------------

# Candidate CSV column names for each context feature
CONTEXT_COLS: dict[str, list[str]] = {
    "Possession": ["Possession"],
    "Shot_Creating_Actions": ["Shot_Creating_Actions", "SCA"],
    "Successful_Dribbles": ["Successful_Dribbles", "Dribbles"],
    "Final_Third_Entries": ["Final_Third_Entries"],
    "Final_Third_Entries_Allowed": ["Final_Third_Entries_Allowed"],
    "Aerial_Battles_Won_Pct": ["Aerial_Battles_Won%", "Aerial_Battles_Won_Pct"],
    "Save_Pct": ["Save%", "Save_Pct"],
    "PPDA": ["PPDA"],
    "Allowed_PPDA": ["Allowed_PPDA"],
}
