"""
Streamlit app for ML Assignment 2 - Classification Model Comparison
Run: streamlit run app.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    from config import (
        TARGET_COLUMN, DATASET_NAME, DATASET_SOURCE,
        DATASET_DESCRIPTION, PROBLEM_TYPE,
    )
except ImportError:
    TARGET_COLUMN    = "target"
    DATASET_NAME     = "Classification Dataset"
    DATASET_SOURCE   = ""
    DATASET_DESCRIPTION = ""
    PROBLEM_TYPE     = "Binary Classification"

MODEL_DIR = os.path.join(ROOT, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "KNN":                 "knn.pkl",
    "Naive Bayes":         "naive_bayes.pkl",
    "Random Forest":       "random_forest.pkl",
}


@st.cache_resource(show_spinner=False)
def load_model(name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_names(model):
    """Return the ordered feature list the pipeline was trained on."""
    return list(model.feature_names_in_)


def preprocess_uploaded(df):
    df = df.copy()

    id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "index")]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    if TARGET_COLUMN in df.columns:
        y_raw = df.pop(TARGET_COLUMN)
        if not pd.api.types.is_numeric_dtype(y_raw):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), name=TARGET_COLUMN)
        else:
            y = y_raw.reset_index(drop=True)
    else:
        y = None

    # Coerce numeric-as-string columns (e.g. Telco TotalCharges has spaces for 0-tenure rows)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        coerced = pd.to_numeric(df[col].str.strip(), errors="coerce")
        if coerced.notna().mean() > 0.8:
            df[col] = coerced

    # Encode remaining categorical columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df, y


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    n_classes = len(np.unique(y))
    avg   = "binary" if n_classes == 2 else "macro"
    multi = "raise"  if n_classes == 2 else "ovr"

    if hasattr(model, "predict_proba"):
        proba  = model.predict_proba(X)
        y_score = proba[:, 1] if n_classes == 2 else proba
    else:
        y_score = model.decision_function(X)

    return {
        "Accuracy":  round(accuracy_score(y, y_pred), 4),
        "AUC":       round(roc_auc_score(y, y_score, multi_class=multi), 4),
        "Precision": round(precision_score(y, y_pred, average=avg, zero_division=0), 4),
        "Recall":    round(recall_score(y, y_pred, average=avg, zero_division=0), 4),
        "F1 Score":  round(f1_score(y, y_pred, average=avg, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y, y_pred), 4),
    }, y_pred


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# page layout

st.set_page_config(
    page_title="ML Assignment 2 - Classification Models",
    layout="wide",
)

st.title("ML Assignment 2 - Classification Model Comparison by Javvadi Venkata Sravan (BITS ID: 2025DA04058)")
st.markdown(f"**Dataset:** {DATASET_NAME} | **Problem type:** {PROBLEM_TYPE}")
if DATASET_SOURCE:
    st.caption(f"Source: {DATASET_SOURCE}")

# sidebar

default_test_file = os.path.join(ROOT, "test_data.csv")
default_exists = os.path.exists(default_test_file)

if "use_default_data" not in st.session_state:
    st.session_state.use_default_data = default_exists
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "uploader_key_index" not in st.session_state:
    st.session_state.uploader_key_index = 0


def clear_default_file():
    st.session_state.use_default_data = False
    st.session_state.selected_file = None
    st.session_state.uploader_key_index += 1

with st.sidebar:
    st.header("Controls")

    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="Upload test_data.csv generated during training.",
        key=f"file_upload_widget_{st.session_state.uploader_key_index}",
    )

    if uploaded_file is not None:
        st.session_state.selected_file = uploaded_file
        st.session_state.use_default_data = False

    if st.session_state.selected_file is not None:
        st.markdown(f"**Current file:** `{st.session_state.selected_file.name}`")
        st.caption("Upload another file to replace it.")
    elif default_exists and st.session_state.use_default_data:
        st.markdown("**Current file:** `test_data.csv` (default)")
        st.caption("This file is loaded automatically from the project root.")
        st.button(
            "Remove default test_data.csv",
            key="remove_default",
            on_click=clear_default_file,
        )
    else:
        st.markdown(
            "No file selected. Upload `test_data.csv` or another compatible CSV file in the sidebar."
        )

    model_choice = st.selectbox(
        "Select model",
        ["All Models"] + list(MODEL_FILES.keys()),
    )

    st.markdown("---")
    st.caption("BITS Pilani M.Tech DSE | Machine Learning")

# main content

tabs = st.tabs(["Results", "Dataset Info", "About"])

# TAB 1: Results

with tabs[0]:
    if st.session_state.selected_file is not None:
        raw_df = pd.read_csv(st.session_state.selected_file)
    elif default_exists and st.session_state.use_default_data:
        raw_df = pd.read_csv(default_test_file)
        st.info("Loaded default test_data.csv from the project root. Upload another CSV to replace it.")
    else:
        raw_df = None

    if raw_df is None:
        st.info("Upload a CSV file from the sidebar to see model predictions and metrics.")
        st.markdown(
            "Expected format: same columns as training data, including the "
            f"`{TARGET_COLUMN}` target column."
        )
    else:
        st.write(f"**Uploaded:** {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

        with st.spinner("Preprocessing..."):
            X_up, y_up = preprocess_uploaded(raw_df)

        if y_up is None:
            st.error(
                f"Target column '{TARGET_COLUMN}' not found in uploaded file. "
                "Please upload a CSV that includes the target column."
            )
        else:
            models_to_eval = (
                list(MODEL_FILES.keys())
                if model_choice == "All Models"
                else [model_choice]
            )

            all_metrics = {}
            all_preds   = {}

            for mname in models_to_eval:
                mdl = load_model(mname)
                if mdl is None:
                    st.warning(f"Model file not found for {mname}. Run python model/train_models.py first.")
                    continue
                try:
                    X_eval = X_up[get_feature_names(mdl)]
                    metrics, y_pred = compute_metrics(mdl, X_eval, y_up)
                    all_metrics[mname] = metrics
                    all_preds[mname]   = y_pred

                    if model_choice != "All Models":
                        st.subheader(f"Results - {mname}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}")
                        col2.metric("AUC",       f"{metrics['AUC']:.4f}")
                        col3.metric("F1 Score",  f"{metrics['F1 Score']:.4f}")
                        col4, col5, col6 = st.columns(3)
                        col4.metric("Precision", f"{metrics['Precision']:.4f}")
                        col5.metric("Recall",    f"{metrics['Recall']:.4f}")
                        col6.metric("MCC",       f"{metrics['MCC']:.4f}")

                        st.markdown("#### Confusion Matrix")
                        fig = plot_confusion_matrix(y_up, y_pred, title=f"{mname} - Confusion Matrix")
                        st.pyplot(fig)
                        plt.close(fig)

                        st.markdown("#### Classification Report")
                        report = classification_report(y_up, y_pred, output_dict=False)
                        st.code(report)

                except Exception as e:
                    st.error(f"Error evaluating {mname}: {e}")

            if model_choice == "All Models" and all_metrics:
                st.subheader("Comparison Table - All Models")
                df_comp = pd.DataFrame(all_metrics).T
                df_comp.index.name = "Model"

                def highlight_max(s):
                    return [
                        "background-color: #d4edda; color: #0f3d03"
                        if v == s.max()
                        else ""
                        for v in s
                    ]

                styled_df = (
                    df_comp.style
                    .apply(highlight_max, axis=0)
                    .format("{:.4f}")
                )

                st.dataframe(styled_df, use_container_width=True)

                best = df_comp["F1 Score"].idxmax()
                st.success(f"Best model by F1 Score: {best} ({df_comp.loc[best, 'F1 Score']:.4f})")

                st.markdown("#### Confusion Matrices")
                cols = st.columns(min(len(all_metrics), 3))
                for i, mname in enumerate(all_metrics):
                    fig = plot_confusion_matrix(y_up, all_preds[mname], title=mname)
                    cols[i % 3].pyplot(fig)
                    plt.close(fig)

# TAB 2: Dataset Info

with tabs[1]:
    st.subheader("Dataset Description")
    st.write(DATASET_DESCRIPTION)
    if DATASET_SOURCE:
        st.markdown(f"**Source:** {DATASET_SOURCE}")
    st.markdown(f"**Target column:** {TARGET_COLUMN}")
    st.markdown(f"**Problem type:** {PROBLEM_TYPE}")

    st.subheader("Models Implemented")
    for m in MODEL_FILES:
        st.markdown(f"- {m}")

# TAB 3: About

with tabs[2]:
    st.markdown("""
### About This App

This Streamlit application was built as part of ML Assignment 2 for
BITS Pilani M.Tech DSE - Machine Learning by Javvadi Venkata Sravan (BITS ID: 2025DA04058)

#### How to use
1. Upload test_data.csv via the sidebar
2. Choose a specific model or select All Models to compare all 5
3. View metrics, confusion matrix, and classification report

#### Models Implemented

| Model | Algorithm |
|-------|-----------|
| Logistic Regression | Linear decision boundary via log-loss |
| Decision Tree | Recursive feature-based splits |
| KNN | k=7 nearest neighbours (scaled) |
| Naive Bayes | Gaussian likelihood assumption |
| Random Forest | 100-tree bagging ensemble |

#### Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, MCC
""")
