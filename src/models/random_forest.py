"""Main entry point for the Premier League predictor."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import build_pipeline
from src.training_common import (
    load_engineered_dataset,
    print_model_report,
    select_feature_columns,
    split_features_and_target,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def main() -> None:
    """Main pipeline: load data, engineer features, train model."""
    data_root = Path(__file__).resolve().parent.parent / "data"
    df = load_engineered_dataset(data_root)
    numeric_cols, cat_cols, all_feat_cols = select_feature_columns(df)

    X_train, y_train, X_val, y_val, X_test, y_test = split_features_and_target(
        df, all_feat_cols
    )

    # ── 11. Train ────────────────────────────────────────────────────────
    model = build_pipeline(numeric_cols, cat_cols)
    model.fit(X_train, y_train)

    # ── 12. Evaluate ─────────────────────────────────────────────────────
    print_model_report(
        model_name="RandomForestClassifier n_estimators=200",
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=len(all_feat_cols),
        include_feature_importance=True,
    )


if __name__ == "__main__":
    main()
