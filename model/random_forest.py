import re
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

TEAM_ALIASES = {
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


def log(msg: str):
    print(f"[INFO] {msg}")


def normalize_team_name(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    if toks and toks[0] in COUNTRY_CODES:
        toks = toks[1:]
    if toks and toks[-1] in COUNTRY_CODES:
        toks = toks[:-1]
    s = " ".join(toks)
    return TEAM_ALIASES.get(s, s)


def build_home_advantage(df: pd.DataFrame) -> pd.Series:
    if "h_a" in df.columns:
        return df["h_a"].astype(str).str.lower().str.strip()
    if "side" in df.columns:
        return df["side"].astype(str).str.lower().str.strip()
    if "Venue" in df.columns:
        return (
            df["Venue"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map({"home": "h", "away": "a"})
        )
    return pd.Series(["a"] * len(df), index=df.index)  # fallback if venue not present


def load_position_map(path: Path) -> dict:
    pos_df = pd.read_csv(path)
    pos_df.columns = [c.strip() for c in pos_df.columns]
    team_col = next(
        (c for c in pos_df.columns if c.lower() in {"team", "teamid", "club"}), None
    )
    pos_col = next(
        (c for c in pos_df.columns if c.lower() in {"position", "pos", "rank"}), None
    )
    if team_col is None or pos_col is None:
        raise ValueError("league_position_after20.csv needs Team and Position columns")
    pos_df["team_norm"] = pos_df[team_col].map(normalize_team_name)
    pos_df["position"] = pd.to_numeric(pos_df[pos_col], errors="coerce")
    pos_df = pos_df.dropna(subset=["team_norm", "position"])
    return dict(zip(pos_df["team_norm"], pos_df["position"]))


def infer_stage(comp: str, stage_text: str, dt: pd.Timestamp) -> str:
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

    # fallback by date if stage column missing
    m = dt.month if pd.notna(dt) else 1
    if comp == "ucl":
        if m in [9, 10, 11, 12]:
            return "group"
        if m in [8]:
            return "late"
        return "knockout"
    if comp == "uel":
        if m in [9, 10, 11, 12]:
            return "group"
        if m in [8]:
            return "late"
        return "knockout"
    if comp == "fa":
        return "late" if m >= 3 else "early"
    if comp == "carabao":
        return "late" if m >= 1 else "early"
    return "early"


def comp_weight(comp: str, stage: str) -> float:
    if comp == "ucl":
        return {"group": 1.0, "knockout": 2.0, "late": 3.0}.get(stage, 1.0)
    if comp == "uel":
        return {"group": 0.8, "knockout": 1.5, "late": 2.0}.get(stage, 0.8)
    if comp == "fa":
        return {"early": 0.5, "late": 1.5}.get(stage, 0.5)
    if comp == "carabao":
        return {"early": 0.3, "late": 1.0}.get(stage, 0.3)
    return 0.0


def load_competition_matches(path: Path, comp: str) -> pd.DataFrame:
    if not path.exists():
        log(f"Skip missing file: {path}")
        return pd.DataFrame(columns=["team", "date", "weight"])

    d = pd.read_csv(path, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    date_col = next((c for c in d.columns if c.lower() == "date"), None)
    home_col = next(
        (c for c in d.columns if c.lower() in {"home team", "home_team", "home"}), None
    )
    away_col = next(
        (c for c in d.columns if c.lower() in {"away team", "away_team", "away"}), None
    )
    stage_col = next((c for c in d.columns if c.lower() in {"stage", "round"}), None)

    if not date_col or not home_col or not away_col:
        log(f"Skip malformed file: {path}")
        return pd.DataFrame(columns=["team", "date", "weight"])

    d["date"] = pd.to_datetime(d[date_col], format="%d/%m/%Y", errors="coerce")
    d["stage_text"] = d[stage_col] if stage_col else ""

    rows = []
    for _, r in d.iterrows():
        if pd.isna(r["date"]):
            continue
        stage = infer_stage(comp, r["stage_text"], r["date"])
        w = comp_weight(comp, stage)
        ht = normalize_team_name(r[home_col])
        at = normalize_team_name(r[away_col])
        rows.append({"team": ht, "date": r["date"], "weight": w})
        rows.append({"team": at, "date": r["date"], "weight": w})

    return pd.DataFrame(rows, columns=["team", "date", "weight"])


def make_match_group_id(df: pd.DataFrame) -> pd.Series:
    # same match => same date + unordered team pair
    t1 = df["Team"].astype(str).str.strip().str.lower()
    t2 = df["Opponent"].astype(str).str.strip().str.lower()
    a = t1.where(t1 <= t2, t2)
    b = t2.where(t1 <= t2, t1)
    d = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return d.fillna("NA") + "|" + a + "|" + b


def group_label_for_strat(g: pd.DataFrame) -> str:
    # draw group => d, else dominant class in group
    vc = g["Result"].value_counts()
    if "d" in vc.index and vc["d"] >= vc.max():
        return "d"
    return vc.idxmax() if not vc.empty else "d"  # type: ignore


def pick_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def main():
    base = Path("/mnt/d/coding/year3/term2/data_science/premier_league_prediction/data")
    f1 = base / "premier_league.csv"
    f2 = base / "additional_data.csv"

    df1 = pd.read_csv(f1, low_memory=False)
    df2 = pd.read_csv(f2, low_memory=False)

    # align identical columns/order
    common_cols = [c for c in df1.columns if c in df2.columns]
    df1 = df1[common_cols].copy()
    df2 = df2[common_cols].copy()

    # merge in memory + deduplicate
    merged = pd.concat([df1, df2], ignore_index=True).drop_duplicates().copy()

    # preprocess
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged["Result"] = merged["Result"].astype(str).str.lower().str.strip()
    merged["Team"] = merged["Team"].map(normalize_team_name)
    merged["Opponent"] = merged["Opponent"].map(normalize_team_name)
    merged["xG"] = pd.to_numeric(merged["xG"], errors="coerce")
    merged["xGA"] = pd.to_numeric(merged["xGA"], errors="coerce")
    merged["home_advantage"] = (
        build_home_advantage(merged).replace({"home": "h", "away": "a"}).fillna("a")
    )

    merged = merged[
        merged["Date"].notna()
        & merged["Result"].isin(["w", "d", "l"])
        & merged["Team"].notna()
        & merged["Opponent"].notna()
    ].copy()

    # NEW: load extra-competition matches and build accumulated fatigue_score
    comp_df = pd.concat(
        [
            load_competition_matches(base / "champion_league.csv", "ucl"),
            load_competition_matches(
                pick_existing(base / "europa_league.csv", base / "europe_league.csv"),
                "uel",
            ),
            load_competition_matches(base / "fa_cup.csv", "fa"),
            load_competition_matches(base / "carabao.csv", "carabao"),
        ],
        ignore_index=True,
    )

    if len(comp_df) == 0:
        merged["fatigue_score"] = 0.0
    else:
        comp_df = (
            comp_df.dropna(subset=["team", "date"]).sort_values(["team", "date"]).copy()
        )
        comp_df["cum_weight"] = comp_df.groupby("team")["weight"].cumsum()

        left = (
            merged.sort_values(["Team", "Date"])
            .reset_index()
            .rename(columns={"index": "_idx"})
        )
        right = comp_df.rename(columns={"team": "Team"}).sort_values(["Team", "date"])

        # cumulative weighted matches played BEFORE each league match date
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
        merged["fatigue_score"] = left.sort_values("_idx")["fatigue_score"].to_numpy()

    # leakage-safe group split by match
    merged["match_group_id"] = make_match_group_id(merged)
    grp = (
        merged.groupby("match_group_id", as_index=False)
        .apply(lambda g: pd.Series({"strat_label": group_label_for_strat(g)}))
        .reset_index(drop=True)
    )

    g_train, g_temp = train_test_split(
        grp,
        test_size=0.30,
        random_state=42,
        stratify=grp["strat_label"],
    )
    g_val, g_test = train_test_split(
        g_temp,
        test_size=1.0 / 3.0,  # 20% val, 10% test
        random_state=42,
        stratify=g_temp["strat_label"],
    )

    train_df = merged[
        merged["match_group_id"].isin(set(g_train["match_group_id"]))
    ].copy()
    val_df = merged[merged["match_group_id"].isin(set(g_val["match_group_id"]))].copy()
    test_df = merged[
        merged["match_group_id"].isin(set(g_test["match_group_id"]))
    ].copy()

    # deterministic shuffle
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # UPDATED: include fatigue_score as feature
    feature_cols = ["xG", "xGA", "home_advantage", "fatigue_score"]
    numeric_features = ["xG", "xGA", "fatigue_score"]
    categorical_features = ["home_advantage"]

    X_train, y_train = train_df[feature_cols], train_df["Result"]
    X_val, y_val = val_df[feature_cols], val_df["Result"]
    X_test, y_test = test_df[feature_cols], test_df["Result"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))


if __name__ == "__main__":
    main()
