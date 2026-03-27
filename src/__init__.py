"""Premier League Match Result Predictor (Style-Aware)

Predicts W / D / L using:
    - Style-aware rolling form : xG, xGA, goals, win-rate, PPDA adjusted for team style
    - Team style classification: possession-heavy, defensive, counter-prone
    - Style-adjusted context   : possession, SCA, dribbles, clean sheets weighted by style
    - Opponent interactions    : style matchup bonuses/penalties
    - Fixture context          : league position, position gap, H2H win rate
    - Squad depth              : rotation rate, impacts fatigue sensitivity
    - Motivation score         : Dynamic, style + fixture aware [0, 1]
    - Referee bias             : Style-sensitive, adjusted for league situation
    - Fatigue                  : cumulative weighted extra-competition load
    - Home advantage           : h / a flag
"""

from .config.config import (
    TEAM_ALIASES,
    COMP_WEIGHTS,
    TEAM_BOOST_OVERRIDES,
    CONTEXT_COLS,
    ROLLING_WINDOW,
)

__all__ = [
    "TEAM_ALIASES",
    "COMP_WEIGHTS",
    "TEAM_BOOST_OVERRIDES",
    "CONTEXT_COLS",
    "ROLLING_WINDOW",
]
