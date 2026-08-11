"""
Central configuration. Edit DATASET_FILE and TARGET_COLUMN once after dropping your CSV.
"""

DATASET_FILE    = "dataset.csv"   # filename in the project root
TARGET_COLUMN   = "Churn"         # change to your dataset's label column (e.g. "y", "target")
RANDOM_STATE    = 42
TEST_SIZE       = 0.20

# Human-readable dataset metadata (used in README / Streamlit UI)
DATASET_NAME        = "Telco Customer Churn"
DATASET_SOURCE      = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
PROBLEM_TYPE        = "Binary Classification"
DATASET_DESCRIPTION = (
    "The Telco Customer Churn dataset contains information about a fictional "
    "telecommunications company. Each row represents one customer; the target "
    "column (Churn) indicates whether the customer left the company within the "
    "last month. The dataset has 7,043 instances and 20 input features covering "
    "demographic data, account information, and services subscribed."
)
