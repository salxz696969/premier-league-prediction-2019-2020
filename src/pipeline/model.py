"""Model training, evaluation, and reporting."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Build sklearn pipeline with preprocessing and RandomForest classifier."""
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
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def print_report(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
) -> None:
    """Print comprehensive model evaluation report."""
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RandomForest  n_estimators=200  features={n_features}")
    print(sep)
    print(f"  Train : {n_train:>6,} rows")
    print(f"  Val   : {n_val:>6,} rows")
    print(f"  Test  : {n_test:>6,} rows")
    print(f"\n  Validation Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy       : {test_acc:.4f}")
    print(f"\nValidation Report:\n{classification_report(y_val, y_val_pred, digits=4)}")
    print(f"Test Report:\n{classification_report(y_test, y_test_pred, digits=4)}")

    # Feature importance
    rf = model.named_steps["clf"]
    feat_names = [
        n.split("__", 1)[-1]
        for n in model.named_steps["preprocess"].get_feature_names_out()
    ]
    importance_df = (
        pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 15 feature importances:")
    print(importance_df.head(15).to_string(index=False))
