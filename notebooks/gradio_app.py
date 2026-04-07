import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "main.ipynb"


def load_notebook_namespace(notebook_path: Path) -> dict[str, Any]:
    """Load imports, constants, and function definitions from the main notebook.

    We intentionally skip execution-heavy cells that train/evaluate full pipelines,
    because this app performs its own leakage-safe per-match training.
    """
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    ns: dict[str, Any] = {}

    skip_tokens = [
        "base_df = load_premier_league(DATA_ROOT)",
        'WDL_LABELS = ["w", "d", "l"]',
        "train_val_raw = (",
        "cv_model_specs = [",
        "comparison_parts = []",
        "Model Accuracy - ",
    ]

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if any(token in source for token in skip_tokens):
            continue
        exec(source, ns)

    required = [
        "load_premier_league",
        "_normalize_base_columns",
        "_add_rolling_helper_columns",
        "_filter_trainable_rows",
        "_load_competition_events",
        "load_position_map",
        "engineer_split_features",
        "select_feature_columns",
        "assert_no_perfect_target_leakage",
        "build_logistic_regression_pipeline",
        "build_decision_tree_pipeline",
        "build_random_forest_pipeline",
        "build_svm_pipeline",
        "build_mlp_pipeline",
        "build_xgboost_pipeline",
        "DATA_ROOT",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Could not load required notebook symbols: {missing}")

    return ns


NS = load_notebook_namespace(NOTEBOOK_PATH)

# User requested no additional data in the app workflow.
NS["USE_ADDITIONAL_DATA"] = False


def prepare_base_rows() -> pd.DataFrame:
    """Load base EPL rows using only data/premier_league.csv (no additional_data)."""
    base_df = NS["load_premier_league"](NS["DATA_ROOT"])
    base_df = NS["_normalize_base_columns"](base_df)
    base_df = NS["_add_rolling_helper_columns"](base_df)
    base_df = NS["_filter_trainable_rows"](base_df)
    base_df = base_df.sort_values("Date").reset_index(drop=True)
    return base_df


BASE_DF = prepare_base_rows()
COMP_DF = NS["_load_competition_events"](NS["DATA_ROOT"])
POS_MAP = NS["load_position_map"](NS["DATA_ROOT"] / "league_position_after20.csv")

MODEL_BUILDERS = {
    "RandomForestClassifier": NS["build_random_forest_pipeline"],
    "XGBoostClassifier": NS["build_xgboost_pipeline"],
    "LogisticRegression": NS["build_logistic_regression_pipeline"],
    "SVM": NS["build_svm_pipeline"],
    "DecisionTreeClassifier": NS["build_decision_tree_pipeline"],
    "MLPClassifier": NS["build_mlp_pipeline"],
}


def get_match_days() -> list[str]:
    return sorted(BASE_DF["Date"].dt.strftime("%Y-%m-%d").unique().tolist())


def matches_for_day(day: str) -> list[str]:
    day_dt = pd.to_datetime(day, errors="coerce")
    if pd.isna(day_dt):
        return []
    rows = BASE_DF[BASE_DF["Date"].dt.normalize() == day_dt.normalize()].copy()
    options = []
    for idx, row in rows.iterrows():
        options.append(
            f"{idx} | {row['Team']} vs {row['Opponent']} | home={str(row['home_advantage']).upper()} | actual={row['Result'].upper()}"
        )
    return options


def update_matches(day: str):
    options = matches_for_day(day)
    value = options[0] if options else None
    return gr.Dropdown(choices=options, value=value)


def parse_match_index(match_option: str) -> int:
    return int(str(match_option).split("|", 1)[0].strip())


def predict_selected_match(day: str, match_option: str, model_name: str):
    if not day or not match_option:
        return "Please select a match day and match.", pd.DataFrame()

    row_idx = parse_match_index(match_option)
    target_row = BASE_DF.iloc[[row_idx]].copy()
    target_date = pd.to_datetime(target_row.iloc[0]["Date"])  # strict date cutoff

    # Leakage guard: training data uses only matches strictly before selected date.
    history_raw = BASE_DF[BASE_DF["Date"] < target_date].copy()

    if len(history_raw) < 50:
        msg = (
            f"Not enough history before {target_date.date()} to train safely "
            f"(rows={len(history_raw)})."
        )
        return msg, pd.DataFrame()

    train_feat = NS["engineer_split_features"](
        history_raw,
        history_df=history_raw,
        comp_df=COMP_DF,
        pos_map=POS_MAP,
    )
    target_feat = NS["engineer_split_features"](
        target_row,
        history_df=history_raw,
        comp_df=COMP_DF,
        pos_map=POS_MAP,
    )

    numeric_cols, cat_cols, feature_cols = NS["select_feature_columns"](train_feat)

    for col in feature_cols:
        if col not in target_feat.columns:
            target_feat[col] = np.nan

    X_train = train_feat[feature_cols]
    y_train = train_feat["Result"]
    X_target = target_feat[feature_cols]

    NS["assert_no_perfect_target_leakage"](X_train, y_train)

    builder = MODEL_BUILDERS[model_name]

    if model_name == "XGBoostClassifier":
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = builder(numeric_cols, cat_cols, n_estimators=200)
        model.fit(X_train, y_train_enc)
        pred_enc = np.asarray(model.predict(X_target), dtype=int)
        pred = le.inverse_transform(pred_enc)[0]

        proba_df = pd.DataFrame(columns=["result", "probability"])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_target)[0]
            labels = le.inverse_transform(np.arange(len(proba), dtype=int))
            proba_df = (
                pd.DataFrame({"result": labels, "probability": proba})
                .sort_values("probability", ascending=False)
                .reset_index(drop=True)
            )
    else:
        model = builder(numeric_cols, cat_cols)
        model.fit(X_train, y_train)
        pred = model.predict(X_target)[0]

        proba_df = pd.DataFrame(columns=["result", "probability"])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_target)[0]
            labels = model.classes_
            proba_df = (
                pd.DataFrame({"result": labels, "probability": proba})
                .sort_values("probability", ascending=False)
                .reset_index(drop=True)
            )

    actual = str(target_row.iloc[0]["Result"])
    info = (
        f"Predicted: {pred.upper()} | Actual: {actual.upper()} | "
        f"History rows used: {len(history_raw):,} | "
        f"Leakage rule: only Date < {target_date.date()}"
    )
    return info, proba_df


def build_app() -> gr.Blocks:
    days = get_match_days()
    initial_day = days[0] if days else None
    initial_matches = matches_for_day(initial_day) if initial_day else []

    with gr.Blocks(title="Premier League Leakage-Safe Match Predictor") as demo:
        gr.Markdown("# Premier League Leakage-Safe Match Predictor")
        gr.Markdown(
            "Select a match day and match from data/premier_league.csv. "
            "The model is trained only on matches before that day."
        )

        with gr.Row():
            day_dd = gr.Dropdown(label="Match Day", choices=days, value=initial_day)
            match_dd = gr.Dropdown(
                label="Match",
                choices=initial_matches,
                value=initial_matches[0] if initial_matches else None,
            )

        model_dd = gr.Dropdown(
            label="Model",
            choices=list(MODEL_BUILDERS.keys()),
            value="RandomForestClassifier",
        )

        run_btn = gr.Button("Predict")
        summary_out = gr.Textbox(label="Prediction Summary")
        probs_out = gr.Dataframe(label="Class Probabilities", interactive=False)

        day_dd.change(fn=update_matches, inputs=[day_dd], outputs=[match_dd])
        run_btn.click(
            fn=predict_selected_match,
            inputs=[day_dd, match_dd, model_dd],
            outputs=[summary_out, probs_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
