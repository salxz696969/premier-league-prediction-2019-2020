"""Clean and match raw EPL data against a clean reference dataset.

Usage example:
    python cleaning_script.py \
      --raw data/unclean_data/epl2020.csv \
      --clean data/premier_league.csv \
      --output-dir data/output \
      --threshold 90
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from typing import Dict, Iterable, List, Optional, Tuple


import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rapidfuzz is required. Install with: pip install rapidfuzz"
    ) from exc


TEAM_NORMALIZATION = {
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "newcastle united": "newcastle",
    "spurs": "tottenham",
    "wolves": "wolverhampton wanderers",
    "sheffield utd": "sheffield united",
}


RAW_TO_STANDARD_COLS = {
    "teamid": "team",
    "matchday": "day",
    "round": "match_day",
    "referee_x": "referee",
    "missed": "conceded",
    "npxgd": "net_npxg",
    "ppda_cal": "ppda",
    "tot_points": "total_points",
    "tot_goal": "total_goals",
    "tot_con": "total_conceded",
    "hs_x": "home_shots",
    "hst_x": "home_sot",
    "hf_x": "home_fouls",
    "hc_x": "home_corners",
    "hy_x": "home_yellow",
    "hr_x": "home_red",
    "as_x": "away_shots",
    "ast_x": "away_sot",
    "af_x": "away_fouls",
    "ac_x": "away_corners",
    "ay_x": "away_yellow",
    "ar_x": "away_red",
}


NUMERIC_COMPARE_COLS = [
    "xg",
    "xga",
    "npxg",
    "npxga",
    "scored",
    "conceded",
    "xpts",
    "net_npxg",
    "ppda",
    "allowed_ppda",
    "total_points",
    "pts",
    "total_goals",
    "total_conceded",
    "home_shots",
    "home_sot",
    "home_fouls",
    "home_corners",
    "home_yellow",
    "home_red",
    "away_shots",
    "away_sot",
    "away_fouls",
    "away_corners",
    "away_yellow",
    "away_red",
]


TEXT_COMPARE_COLS = ["day", "date", "referee", "team", "opponent", "result"]


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_team_name(value: object) -> object:
    text = normalize_text(value)
    if pd.isna(text):
        return text
    return TEAM_NORMALIZATION.get(str(text), text)


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    unnamed = [c for c in out.columns if c.startswith("unnamed") or c == ""]
    if unnamed:
        out = out.drop(columns=unnamed)
    return out


def parse_date_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def standardize_clean_dataframe(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe_columns(clean_df)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(normalize_text)

    if "date" in df.columns:
        df["date"] = parse_date_column(df["date"])

    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_team_name)

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(normalize_team_name)

    df = coerce_numeric(df, NUMERIC_COMPARE_COLS + ["match_day"])

    return df.drop_duplicates().reset_index(drop=True)


def derive_opponent(raw_df: pd.DataFrame) -> pd.Series:
    fixture_cols = [
        c
        for c in ["date", "match_day", "referee", "home_shots", "home_sot", "away_shots", "away_sot"]
        if c in raw_df.columns
    ]

    if not fixture_cols or "team" not in raw_df.columns:
        return pd.Series([pd.NA] * len(raw_df), index=raw_df.index)

    opponents = pd.Series([pd.NA] * len(raw_df), index=raw_df.index)
    grouped = raw_df.groupby(fixture_cols, dropna=False, sort=False)

    for _, idx in grouped.groups.items():
        indices = list(idx)
        if len(indices) != 2:
            continue
        team_a = raw_df.loc[indices[0], "team"]
        team_b = raw_df.loc[indices[1], "team"]
        opponents.loc[indices[0]] = team_b
        opponents.loc[indices[1]] = team_a

    return opponents


def standardize_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe_columns(raw_df)
    df = df.rename(columns=RAW_TO_STANDARD_COLS)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(normalize_text)

    if "date" in df.columns:
        df["date"] = parse_date_column(df["date"])

    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_team_name)

    df["opponent"] = derive_opponent(df)
    df["opponent"] = df["opponent"].map(normalize_team_name)

    df = coerce_numeric(df, NUMERIC_COMPARE_COLS + ["match_day"])

    if "result" in df.columns:
        df["result"] = df["result"].replace({"w": "w", "l": "l", "d": "d"})

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    text_cols = [c for c in out.columns if c not in numeric_cols]

    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())

    for col in text_cols:
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else ""
        out[col] = out[col].fillna(fill_value)

    return out


def build_exact_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["date", "match_day", "team"]:
        if col in df.columns:
            parts.append(df[col].astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))

    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def fuzzy_best_match(
    raw_row: pd.Series,
    candidates: pd.DataFrame,
    team_weight: float = 0.7,
    referee_weight: float = 0.3,
) -> Tuple[Optional[int], float]:
    if candidates.empty:
        return None, 0.0

    best_idx = None
    best_score = -1.0
    raw_team = str(raw_row.get("team", ""))
    raw_ref = str(raw_row.get("referee", ""))

    for idx, row in candidates.iterrows():
        team_score = fuzz.token_sort_ratio(raw_team, str(row.get("team", "")))
        ref_score = fuzz.token_sort_ratio(raw_ref, str(row.get("referee", "")))
        score = team_weight * team_score + referee_weight * ref_score

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, float(best_score)


def match_raw_to_clean(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    threshold: float,
    high_conf_threshold: float,
) -> pd.DataFrame:
    raw = raw_df.copy()
    clean = clean_df.copy()

    raw["exact_key"] = build_exact_key(raw)
    clean["exact_key"] = build_exact_key(clean)

    clean_exact_map = clean.reset_index().set_index("exact_key")["index"].to_dict()

    matched_clean_index: List[Optional[int]] = []
    matched_by: List[str] = []
    confidence: List[float] = []
    status: List[str] = []

    used_clean_indices = set()

    for _, row in raw.iterrows():
        key = row["exact_key"]
        if key in clean_exact_map:
            idx = clean_exact_map[key]
            if idx not in used_clean_indices:
                used_clean_indices.add(idx)
                matched_clean_index.append(idx)
                matched_by.append("exact")
                confidence.append(100.0)
                status.append("matched")
                continue

        if "date" in clean.columns and "date" in raw.columns:
            candidates = clean[clean["date"] == row.get("date")]
        else:
            candidates = clean

        candidates = candidates.loc[~candidates.index.isin(used_clean_indices)]
        best_idx, score = fuzzy_best_match(row, candidates)

        if best_idx is None:
            matched_clean_index.append(None)
            matched_by.append("none")
            confidence.append(0.0)
            status.append("unmatched")
            continue

        if score >= threshold:
            used_clean_indices.add(best_idx)
            matched_clean_index.append(best_idx)
            matched_by.append("fuzzy")
            confidence.append(score)
            if score >= high_conf_threshold:
                status.append("matched")
            else:
                status.append("low_confidence")
        else:
            matched_clean_index.append(None)
            matched_by.append("none")
            confidence.append(score)
            status.append("unmatched")

    raw["matched_clean_index"] = matched_clean_index
    raw["match_method"] = matched_by
    raw["match_confidence"] = confidence
    raw["match_status"] = status

    clean_prefixed = clean.reset_index().add_prefix("clean_")
    merged = raw.merge(
        clean_prefixed,
        left_on="matched_clean_index",
        right_on="clean_index",
        how="left",
    )

    return merged


def compute_validation_metrics(matched_df: pd.DataFrame) -> Dict[str, float]:
    total = len(matched_df)
    matched_count = int((matched_df["match_status"] == "matched").sum())
    low_conf_count = int((matched_df["match_status"] == "low_confidence").sum())
    unmatched_count = int((matched_df["match_status"] == "unmatched").sum())

    field_scores: List[float] = []
    compare_cols = NUMERIC_COMPARE_COLS + TEXT_COMPARE_COLS

    for _, row in matched_df[matched_df["match_status"] != "unmatched"].iterrows():
        agreements = []
        for col in compare_cols:
            clean_col = f"clean_{col}"
            if col not in matched_df.columns or clean_col not in matched_df.columns:
                continue

            raw_val = row[col]
            clean_val = row[clean_col]

            if pd.isna(raw_val) and pd.isna(clean_val):
                agreements.append(1.0)
                continue

            if pd.api.types.is_number(raw_val) and pd.api.types.is_number(clean_val):
                agreements.append(1.0 if abs(float(raw_val) - float(clean_val)) <= 1e-6 else 0.0)
            else:
                agreements.append(1.0 if str(raw_val) == str(clean_val) else 0.0)

        if agreements:
            field_scores.append(sum(agreements) / len(agreements))

    field_accuracy = float(sum(field_scores) / len(field_scores) * 100) if field_scores else 0.0

    return {
        "total_raw_records": total,
        "matched_records": matched_count,
        "low_confidence_records": low_conf_count,
        "unmatched_records": unmatched_count,
        "match_coverage_pct": (matched_count / total * 100) if total else 0.0,
        "field_accuracy_pct": field_accuracy,
    }


def print_schema_differences(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    raw_cols = set(raw_df.columns)
    clean_cols = set(clean_df.columns)

    print("\n=== Step 1: Dataset Structure ===")
    print(f"Raw shape:   {raw_df.shape}")
    print(f"Clean shape: {clean_df.shape}")

    only_raw = sorted(raw_cols - clean_cols)
    only_clean = sorted(clean_cols - raw_cols)

    print("\nColumns only in raw:", only_raw)
    print("\nColumns only in clean:", only_clean)


def save_outputs(matched_df: pd.DataFrame, metrics: Dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_out = matched_df[matched_df["match_status"] == "matched"]
    unmatched_out = matched_df[matched_df["match_status"] == "unmatched"]
    low_conf_out = matched_df[matched_df["match_status"] == "low_confidence"]

    matched_path = output_dir / "cleaned_matched_dataset.csv"
    unmatched_path = output_dir / "unmatched_records.csv"
    low_conf_path = output_dir / "low_confidence_matches.csv"
    report_path = output_dir / "matching_report.json"

    matched_out.to_csv(matched_path, index=False)
    unmatched_out.to_csv(unmatched_path, index=False)
    low_conf_out.to_csv(low_conf_path, index=False)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Step 7: Outputs Saved ===")
    print(f"Matched dataset:      {matched_path}")
    print(f"Unmatched records:    {unmatched_path}")
    print(f"Low confidence cases: {low_conf_path}")
    print(f"Validation report:    {report_path}")


def run_pipeline(
    raw_path: Path,
    clean_path: Path,
    output_dir: Path,
    threshold: float,
    high_conf_threshold: float,
) -> None:
    # Step 1: Load datasets
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(clean_path)

    print_schema_differences(raw_df, clean_df)

    # Step 2 + Step 3: Standardize and clean
    raw_std = standardize_raw_dataframe(raw_df)
    clean_std = standardize_clean_dataframe(clean_df)

    raw_std = fill_missing_values(raw_std)
    clean_std = fill_missing_values(clean_std)

    # Step 4 + Step 5: Match and merge
    print("\n=== Step 4 & 5: Matching ===")
    matched = match_raw_to_clean(raw_std, clean_std, threshold, high_conf_threshold)

    # Step 6: Validation
    metrics = compute_validation_metrics(matched)
    print("\n=== Step 6: Validation ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    # Step 7: Output
    save_outputs(matched, metrics, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and match EPL raw data to clean reference")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/unclean_data/epl2020.csv"),
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=Path("data/premier_league.csv"),
        help="Path to clean reference dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory to save output files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Fuzzy-match threshold for accepting a candidate",
    )
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=95.0,
        help="Threshold above which fuzzy matches are considered high confidence",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        raw_path=args.raw,
        clean_path=args.clean,
        output_dir=args.output_dir,
        threshold=args.threshold,
        high_conf_threshold=args.high_conf_threshold,
    )


if __name__ == "__main__":
    main()

