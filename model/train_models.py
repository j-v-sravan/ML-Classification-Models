"""
Trains all 5 classification models, evaluates with 6 metrics each,
saves trained pipelines as .pkl files, and exports test_data.csv.

Run from project root:
    python model/train_models.py
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import DATASET_FILE, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_preprocess():
    path = os.path.join(ROOT, DATASET_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place dataset.csv in the project root and update config.py if needed."
        )
    df = pd.read_csv(path)
    print(f"Loaded: {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    # Drop ID columns
    id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "index")]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        print(f"Dropped ID columns: {id_cols}")

    # Coerce numeric-as-string columns (Telco TotalCharges has spaces for new customers)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col == TARGET_COLUMN:
            continue
        coerced = pd.to_numeric(df[col].str.strip(), errors="coerce")
        if coerced.notna().mean() > 0.8:
            df[col] = coerced

    # Encode categorical features
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col == TARGET_COLUMN:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Encode target
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        le = LabelEncoder()
        df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN].astype(str))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"Features after preprocessing: {X.shape[1]} | Target classes: {sorted(y.unique())}")
    return X, y


def build_pipelines():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "macro"
    multi = "raise" if n_classes == 2 else "ovr"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        y_score = proba[:, 1] if n_classes == 2 else proba
    else:
        y_score = model.decision_function(X_test)

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_score, multi_class=multi), 4),
        "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }


def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics

        safe_name = name.replace(" ", "_").lower()
        pkl_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline, f)

        print(f"[OK] {name:25s}  Acc={metrics['Accuracy']:.4f}  AUC={metrics['AUC']:.4f}"
              f"  F1={metrics['F1 Score']:.4f}  MCC={metrics['MCC']:.4f}  -> {pkl_path}")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    df_res = pd.DataFrame(results).T
    df_res.index.name = "Model"
    print(df_res.to_string())
    print("=" * 80)

    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    df_res.to_csv(metrics_path)
    print(f"\nMetrics  -> {metrics_path}")

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_csv = os.path.join(ROOT, "test_data.csv")
    test_df.to_csv(test_csv, index=False)
    print(f"Test CSV -> {test_csv}")


if __name__ == "__main__":
    main()
