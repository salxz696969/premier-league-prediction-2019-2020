from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent / "data" / "premier_league.csv"
TARGET_COL = "Result"
ROLLING_WINDOW = 5

BASE_NUMERIC_FEATURES = [
    "rolling_xG_5",
    "rolling_xGA_5",
    "rolling_ppda_5",
    "rolling_allowed_ppda_5",
    "rolling_possession_5",
    "rolling_win_rate_5",
]
BASE_CATEGORICAL_FEATURES = ["Team", "Opponent", "Referee", "Formation"]

GOAL_BLOCKLIST_RAW = {
    "scored",
    "conceded",
    "goals",
    "goals_for",
    "goals_against",
    "home_goals",
    "away_goals",
    "gf",
    "ga",
    "xpts",
    "xg",
    "xga",
    "npxg",
    "psxg",
    "result",
    "pts",
    "points",
    "save%",
    "pk_allowed",
    "og",
    "away_red",
    "home_red",
    "net_npxg",
    "total_points",
    "total_goals",
    "total_conceded",
}


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["xG", "xGA", "PPDA", "Allowed_PPDA", "Possession"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = [TARGET_COL, "Date", "Team", "Opponent", "Referee", "Formation"]
    df = df.dropna(subset=needed).copy()
    df = df[df[TARGET_COL].isin(["w", "d", "l"])].copy()
    return df.sort_values("Date").reset_index(drop=True)


def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_win_val"] = out[TARGET_COL].map({"w": 1.0, "d": 0.5, "l": 0.0})
    out = out.sort_values(["Team", "Date"]).copy()
    grp = out.groupby("Team", sort=False)

    out[f"rolling_xG_{ROLLING_WINDOW}"] = grp["xG"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_xGA_{ROLLING_WINDOW}"] = grp["xGA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_ppda_{ROLLING_WINDOW}"] = grp["PPDA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_allowed_ppda_{ROLLING_WINDOW}"] = grp["Allowed_PPDA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_possession_{ROLLING_WINDOW}"] = grp["Possession"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_win_rate_{ROLLING_WINDOW}"] = grp["_win_val"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )

    out.drop(columns=["_win_val"], inplace=True)
    return out.sort_values("Date").reset_index(drop=True)


def _time_based_split(
    df: pd.DataFrame, val_frac: float = 0.20, test_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("Date").reset_index(drop=True)
    unique_dates = np.array(sorted(ordered["Date"].dropna().unique()))
    n_dates = len(unique_dates)
    if n_dates < 10:
        raise ValueError("Not enough unique dates for time-based split.")

    n_test = max(1, int(round(n_dates * test_frac)))
    n_val = max(1, int(round(n_dates * val_frac)))

    test_dates = set(unique_dates[-n_test:])
    val_dates = set(unique_dates[-(n_test + n_val) : -n_test])
    train_dates = set(unique_dates[: -(n_test + n_val)])

    train_df = ordered[ordered["Date"].isin(train_dates)].copy().reset_index(drop=True)
    val_df = ordered[ordered["Date"].isin(val_dates)].copy().reset_index(drop=True)
    test_df = ordered[ordered["Date"].isin(test_dates)].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def _build_current_match_feature_columns(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    base_numeric_cols: list[str],
    base_cat_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    excluded = {TARGET_COL}
    base_cols = set(base_numeric_cols + base_cat_cols)
    goal_blocklist = {s.lower() for s in GOAL_BLOCKLIST_RAW}

    extra_numeric: list[str] = []
    extra_cat: list[str] = []
    shared_cols = [c for c in train_feat.columns if c in test_feat.columns]

    for col in shared_cols:
        col_l = col.lower().strip()
        if col in excluded or col in base_cols:
            continue
        if col_l in goal_blocklist or col_l.startswith("_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(train_feat[col]):
            continue
        if pd.api.types.is_numeric_dtype(train_feat[col]) and pd.api.types.is_numeric_dtype(
            test_feat[col]
        ):
            extra_numeric.append(col)
            continue
        if (
            pd.api.types.is_object_dtype(train_feat[col])
            or pd.api.types.is_string_dtype(train_feat[col])
            or pd.api.types.is_bool_dtype(train_feat[col])
        ):
            extra_cat.append(col)

    numeric_cols = base_numeric_cols + sorted(extra_numeric)
    cat_cols = base_cat_cols + sorted(extra_cat)
    all_cols = numeric_cols + cat_cols
    return numeric_cols, cat_cols, all_cols


def _build_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", xgb)])


def _train_model(
    df: pd.DataFrame,
) -> tuple[Pipeline, LabelEncoder, float, str, pd.DataFrame, list[str]]:
    feat_df = _build_rolling_features(df)
    train_df, val_df, test_df = _time_based_split(feat_df, val_frac=0.20, test_frac=0.20)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True).sort_values("Date")

    numeric_cols, cat_cols, all_cols = _build_current_match_feature_columns(
        train_feat=train_val_df,
        test_feat=test_df,
        base_numeric_cols=BASE_NUMERIC_FEATURES,
        base_cat_cols=BASE_CATEGORICAL_FEATURES,
    )

    for col in all_cols:
        if col not in train_val_df.columns:
            train_val_df[col] = np.nan
        if col not in test_df.columns:
            test_df[col] = np.nan

    X_train = train_val_df[all_cols].copy()
    y_train = train_val_df[TARGET_COL].copy()
    X_test = test_df[all_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    label_encoder = LabelEncoder()
    y_train_enc = np.asarray(label_encoder.fit_transform(y_train), dtype=int)
    y_test_enc = np.asarray(label_encoder.transform(y_test), dtype=int)

    pipeline = _build_pipeline(numeric_cols, cat_cols)
    pipeline.fit(X_train, y_train_enc)

    y_pred_enc = np.asarray(pipeline.predict(X_test), dtype=int)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    test_acc = float(accuracy_score(y_test_enc, y_pred_enc))
    report = str(classification_report(y_test, y_pred, digits=4))
    return pipeline, label_encoder, test_acc, report, feat_df, all_cols


DF = _load_data()
PIPELINE, LABEL_ENCODER, TEST_ACCURACY, TEST_REPORT, FEAT_DF, MODEL_FEATURES = _train_model(
    DF
)


def _match_option_label(row: pd.Series, idx: int) -> str:
    date_str = (
        row["Date"].strftime("%Y-%m-%d")
        if pd.notna(row["Date"])
        else "unknown-date"
    )
    return f"{idx:04d} | {date_str} | {row['Team']} vs {row['Opponent']} ({row[TARGET_COL]})"


MATCH_OPTIONS: dict[str, int] = {
    _match_option_label(FEAT_DF.iloc[i], i): i for i in range(len(FEAT_DF))
}


def predict_selected_match(selected_option: str):
    idx = MATCH_OPTIONS[selected_option]
    row = FEAT_DF.iloc[idx].copy()

    x_row = FEAT_DF.iloc[[idx]].copy()
    for col in MODEL_FEATURES:
        if col not in x_row.columns:
            x_row[col] = np.nan
    x_row = x_row[MODEL_FEATURES]

    pred_enc = int(np.asarray(PIPELINE.predict(x_row), dtype=int)[0])
    pred_label = LABEL_ENCODER.inverse_transform([pred_enc])[0]

    proba = PIPELINE.predict_proba(x_row)[0]
    classes = LABEL_ENCODER.inverse_transform(np.arange(len(proba)))
    proba_df = (
        pd.DataFrame({"class": classes, "probability": proba})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    top_prob = float(cast(float, proba_df.loc[0, "probability"]))

    pretty_result = {"w": "Win", "d": "Draw", "l": "Loss"}.get(pred_label, pred_label)
    match_date = (
        row["Date"].strftime("%Y-%m-%d")
        if pd.notna(row["Date"])
        else "unknown-date"
    )

    summary_md = (
        f"### Prediction: **{pretty_result}** (`{pred_label}`)\n"
        f"- Confidence: **{top_prob:.2%}**\n"
        f"- Match: **{row['Team']} vs {row['Opponent']}**\n"
        f"- Date: **{match_date}**\n"
        f"- Actual label in dataset: **{row[TARGET_COL]}**"
    )

    raw_view_cols = [
        "Date",
        "Team",
        "Opponent",
        "Referee",
        "Formation",
        "xG",
        "xGA",
        "rolling_xG_5",
        "rolling_xGA_5",
        "PPDA",
        "Allowed_PPDA",
        "Possession",
        TARGET_COL,
    ]
    raw_row_df = FEAT_DF.iloc[[idx]][raw_view_cols].copy()
    return summary_md, proba_df, raw_row_df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Premier League XGBoost Demo") as app:
        gr.Markdown("# Premier League Match Prediction Demo (XGBoost)")
        gr.Markdown(
            "v2-style setup: chronological split, rolling-xG features, and v2 blocklist in feature builder."
        )

        with gr.Row():
            with gr.Column(scale=2):
                match_dropdown = gr.Dropdown(
                    choices=list(MATCH_OPTIONS.keys()),
                    value=list(MATCH_OPTIONS.keys())[0],
                    label="Pick a Match Row",
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown(
                    f"### Model Snapshot\n"
                    f"- Rows used: **{len(FEAT_DF):,}**\n"
                    f"- Features used by model: **{len(MODEL_FEATURES)}**\n"
                    f"- Includes rolling: **rolling_xG_5 / rolling_xGA_5**\n"
                    f"- Test accuracy: **{TEST_ACCURACY:.4f}**"
                )

        prediction_md = gr.Markdown(label="Prediction Summary")
        with gr.Row():
            proba_table = gr.Dataframe(
                label="Predicted Class Probabilities",
                interactive=False,
                wrap=True,
            )
            raw_table = gr.Dataframe(
                label="Selected Match Data",
                interactive=False,
                wrap=True,
            )

        with gr.Accordion("Full classification report (test split)", open=False):
            gr.Textbox(value=TEST_REPORT, label="Classification Report", lines=14)

        predict_btn.click(
            fn=predict_selected_match,
            inputs=[match_dropdown],
            outputs=[prediction_md, proba_table, raw_table],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent / "data" / "premier_league.csv"
TARGET_COL = "Result"
ROLLING_WINDOW = 5

# v2-style high-signal base columns that include rolling xG/xGA.
BASE_NUMERIC_FEATURES = [
    "rolling_xG_5",
    "rolling_xGA_5",
    "rolling_ppda_5",
    "rolling_allowed_ppda_5",
    "rolling_possession_5",
    "rolling_win_rate_5",
]
BASE_CATEGORICAL_FEATURES = ["Team", "Opponent", "Referee", "Formation"]

# Same leakage blocklist requested from v2 notebook.
GOAL_BLOCKLIST_RAW = {
    "scored",
    "conceded",
    "goals",
    "goals_for",
    "goals_against",
    "home_goals",
    "away_goals",
    "gf",
    "ga",
    "xpts",
    "xg",
    "xga",
    "npxg",
    "psxg",
    "result",
    "pts",
    "points",
    "save%",
    "pk_allowed",
    "og",
    "away_red",
    "home_red",
    "net_npxg",
    "total_points",
    "total_goals",
    "total_conceded",
}


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["xG", "xGA", "PPDA", "Allowed_PPDA", "Possession"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = [TARGET_COL, "Date", "Team", "Opponent", "Referee", "Formation"]
    df = df.dropna(subset=needed).copy()
    df = df[df[TARGET_COL].isin(["w", "d", "l"])].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_win_val"] = out[TARGET_COL].map({"w": 1.0, "d": 0.5, "l": 0.0})

    out = out.sort_values(["Team", "Date"]).copy()
    grp = out.groupby("Team", sort=False)

    out[f"rolling_xG_{ROLLING_WINDOW}"] = grp["xG"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_xGA_{ROLLING_WINDOW}"] = grp["xGA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_ppda_{ROLLING_WINDOW}"] = grp["PPDA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_allowed_ppda_{ROLLING_WINDOW}"] = grp["Allowed_PPDA"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_possession_{ROLLING_WINDOW}"] = grp["Possession"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )
    out[f"rolling_win_rate_{ROLLING_WINDOW}"] = grp["_win_val"].transform(
        lambda s: _lagged_rolling_mean(s, ROLLING_WINDOW)
    )

    out.drop(columns=["_win_val"], inplace=True)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _time_based_split(
    df: pd.DataFrame, val_frac: float = 0.20, test_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("Date").reset_index(drop=True)
    unique_dates = np.array(sorted(ordered["Date"].dropna().unique()))
    n_dates = len(unique_dates)
    if n_dates < 10:
        raise ValueError("Not enough unique dates for time-based split.")

    n_test = max(1, int(round(n_dates * test_frac)))
    n_val = max(1, int(round(n_dates * val_frac)))

    test_dates = set(unique_dates[-n_test:])
    val_dates = set(unique_dates[-(n_test + n_val) : -n_test])
    train_dates = set(unique_dates[: -(n_test + n_val)])

    train_df = ordered[ordered["Date"].isin(train_dates)].copy().reset_index(drop=True)
    val_df = ordered[ordered["Date"].isin(val_dates)].copy().reset_index(drop=True)
    test_df = ordered[ordered["Date"].isin(test_dates)].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def _build_current_match_feature_columns(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    base_numeric_cols: list[str],
    base_cat_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    excluded = {TARGET_COL}
    base_cols = set(base_numeric_cols + base_cat_cols)
    goal_blocklist = {s.lower() for s in GOAL_BLOCKLIST_RAW}

    extra_numeric: list[str] = []
    extra_cat: list[str] = []
    shared_cols = [c for c in train_feat.columns if c in test_feat.columns]

    for col in shared_cols:
        col_l = col.lower().strip()
        if col in excluded or col in base_cols:
            continue
        if col_l in goal_blocklist or col_l.startswith("_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(train_feat[col]):
            continue
        if pd.api.types.is_numeric_dtype(train_feat[col]) and pd.api.types.is_numeric_dtype(
            test_feat[col]
        ):
            extra_numeric.append(col)
            continue
        if (
            pd.api.types.is_object_dtype(train_feat[col])
            or pd.api.types.is_string_dtype(train_feat[col])
            or pd.api.types.is_bool_dtype(train_feat[col])
        ):
            extra_cat.append(col)

    numeric_cols = base_numeric_cols + sorted(extra_numeric)
    cat_cols = base_cat_cols + sorted(extra_cat)
    all_cols = numeric_cols + cat_cols
    return numeric_cols, cat_cols, all_cols


def _build_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", xgb)])


def _train_model(
    df: pd.DataFrame,
) -> tuple[Pipeline, LabelEncoder, float, str, pd.DataFrame, list[str]]:
    feat_df = _build_rolling_features(df)
    train_df, val_df, test_df = _time_based_split(feat_df, val_frac=0.20, test_frac=0.20)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True).sort_values("Date")

    numeric_cols, cat_cols, all_cols = _build_current_match_feature_columns(
        train_feat=train_val_df,
        test_feat=test_df,
        base_numeric_cols=BASE_NUMERIC_FEATURES,
        base_cat_cols=BASE_CATEGORICAL_FEATURES,
    )

    for col in all_cols:
        if col not in train_val_df.columns:
            train_val_df[col] = np.nan
        if col not in test_df.columns:
            test_df[col] = np.nan

    X_train = train_val_df[all_cols].copy()
    y_train = train_val_df[TARGET_COL].copy()
    X_test = test_df[all_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    label_encoder = LabelEncoder()
    y_train_enc = np.asarray(label_encoder.fit_transform(y_train), dtype=int)
    y_test_enc = np.asarray(label_encoder.transform(y_test), dtype=int)

    pipeline = _build_pipeline(numeric_cols, cat_cols)
    pipeline.fit(X_train, y_train_enc)

    y_pred_enc = np.asarray(pipeline.predict(X_test), dtype=int)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    test_acc = float(accuracy_score(y_test_enc, y_pred_enc))
    report = str(classification_report(y_test, y_pred, digits=4))
    return pipeline, label_encoder, test_acc, report, feat_df, all_cols


DF = _load_data()
PIPELINE, LABEL_ENCODER, TEST_ACCURACY, TEST_REPORT, FEAT_DF, MODEL_FEATURES = _train_model(
    DF
)


def _match_option_label(row: pd.Series, idx: int) -> str:
    date_str = (
        row["Date"].strftime("%Y-%m-%d")
        if pd.notna(row["Date"])
        else "unknown-date"
    )
    return f"{idx:04d} | {date_str} | {row['Team']} vs {row['Opponent']} ({row[TARGET_COL]})"


MATCH_OPTIONS: dict[str, int] = {
    _match_option_label(FEAT_DF.iloc[i], i): i for i in range(len(FEAT_DF))
}


def predict_selected_match(selected_option: str):
    idx = MATCH_OPTIONS[selected_option]
    row = FEAT_DF.iloc[idx].copy()

    x_row = FEAT_DF.iloc[[idx]].copy()
    for col in MODEL_FEATURES:
        if col not in x_row.columns:
            x_row[col] = np.nan
    x_row = x_row[MODEL_FEATURES]

    pred_enc = int(np.asarray(PIPELINE.predict(x_row), dtype=int)[0])
    pred_label = LABEL_ENCODER.inverse_transform([pred_enc])[0]

    proba = PIPELINE.predict_proba(x_row)[0]
    classes = LABEL_ENCODER.inverse_transform(np.arange(len(proba)))
    proba_df = (
        pd.DataFrame({"class": classes, "probability": proba})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    top_prob = float(cast(float, proba_df.loc[0, "probability"]))

    pretty_result = {"w": "Win", "d": "Draw", "l": "Loss"}.get(pred_label, pred_label)
    match_date = (
        row["Date"].strftime("%Y-%m-%d")
        if pd.notna(row["Date"])
        else "unknown-date"
    )

    summary_md = (
        f"### Prediction: **{pretty_result}** (`{pred_label}`)\n"
        f"- Confidence: **{top_prob:.2%}**\n"
        f"- Match: **{row['Team']} vs {row['Opponent']}**\n"
        f"- Date: **{match_date}**\n"
        f"- Actual label in dataset: **{row[TARGET_COL]}**"
    )

    raw_view_cols = [
        "Date",
        "Team",
        "Opponent",
        "Referee",
        "Formation",
        "xG",
        "xGA",
        "rolling_xG_5",
        "rolling_xGA_5",
        "PPDA",
        "Allowed_PPDA",
        "Possession",
        TARGET_COL,
    ]
    raw_row_df = FEAT_DF.iloc[[idx]][raw_view_cols].copy()
    return summary_md, proba_df, raw_row_df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Premier League XGBoost Demo") as app:
        gr.Markdown("# Premier League Match Prediction Demo (XGBoost)")
        gr.Markdown(
            "v2-style setup: chronological split, rolling-xG features, and v2 blocklist in feature builder."
        )

        with gr.Row():
            with gr.Column(scale=2):
                match_dropdown = gr.Dropdown(
                    choices=list(MATCH_OPTIONS.keys()),
                    value=list(MATCH_OPTIONS.keys())[0],
                    label="Pick a Match Row",
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown(
                    f"### Model Snapshot\n"
                    f"- Rows used: **{len(FEAT_DF):,}**\n"
                    f"- Features used by model: **{len(MODEL_FEATURES)}**\n"
                    f"- Includes rolling: **rolling_xG_5 / rolling_xGA_5**\n"
                    f"- Test accuracy: **{TEST_ACCURACY:.4f}**"
                )

        prediction_md = gr.Markdown(label="Prediction Summary")
        with gr.Row():
            proba_table = gr.Dataframe(
                label="Predicted Class Probabilities",
                interactive=False,
                wrap=True,
            )
            raw_table = gr.Dataframe(
                label="Selected Match Data",
                interactive=False,
                wrap=True,
            )

        with gr.Accordion("Full classification report (test split)", open=False):
            gr.Textbox(value=TEST_REPORT, label="Classification Report", lines=14)

        predict_btn.click(
            fn=predict_selected_match,
            inputs=[match_dropdown],
            outputs=[prediction_md, proba_table, raw_table],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
