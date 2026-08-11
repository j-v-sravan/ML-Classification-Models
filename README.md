# ML Assignment 2 - Classification Model Comparison by Javvadi Venkata Sravan (BITS ID: 2025DA04058)

**Course:** Machine Learning | BITS Pilani M.Tech DSE

---

## a. Problem Statement

Build and compare five classification models on a real-world dataset.
For each model, compute six evaluation metrics and identify the best-performing
model. Deploy the solution as an interactive Streamlit web application.

---

## b. Dataset Description

| Property | Value |
|----------|-------|
| **Name** | Telco Customer Churn |
| **Source** | Kaggle - blastchar/telco-customer-churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Instances** | 7,043 |
| **Features** | 20 input features |
| **Target** | Churn (binary: Yes / No) |
| **Problem type** | Binary Classification |

The dataset describes customers of a fictional telecom company.
Each row represents one customer. Features cover demographics
(gender, senior citizen status, dependents), account information
(tenure, contract type, payment method, monthly/total charges),
and services subscribed (phone, internet, streaming, security).
The target variable indicates whether the customer cancelled their
subscription within the last month.

---

## c. GitHub Repository Link

**https://github.com/j-v-sravan/ml-classification-models**

Repository structure:

```
ml-assignment-2/
    app.py                   - Streamlit web application
    config.py                - Dataset and target column configuration
    requirements.txt
    README.md
    test_data.csv            - Held-out test split (20% of dataset)
    model/
        train_models.py      - Training and evaluation script
        logistic_regression.pkl
        decision_tree.pkl
        knn.pkl
        naive_bayes.pkl
        random_forest.pkl
        metrics.csv
```

---

## d. Models Used and Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8070 | 0.8416 | 0.6584 | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree | 0.7559 | 0.7587 | 0.5391 | 0.5535 | 0.5462 | 0.3793 |
| KNN | 0.7594 | 0.7847 | 0.5499 | 0.5160 | 0.5324 | 0.3710 |
| Naive Bayes | 0.6558 | 0.8096 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Random Forest (Ensemble) | 0.7871 | 0.8215 | 0.6267 | 0.4893 | 0.5495 | 0.4183 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer (Accuracy 0.8070, F1 0.6092, MCC 0.4843). Highest accuracy and best F1 among all models, indicating that the churn decision boundary is well-captured by a linear model after feature scaling. The AUC of 0.8416 confirms strong rank-ordering of churn probability. Balanced Precision (0.6584) and Recall (0.5668) make it the most reliable model for this dataset. |
| Decision Tree | Second weakest performer (Accuracy 0.7559, AUC 0.7587). The lowest AUC in the comparison signals overfitting - the tree memorises training splits rather than generalising. Axis-aligned decision boundaries struggle with the mix of binary and continuous features in this dataset. Increasing max_depth could improve recall but risks further overfitting. |
| KNN | Weakest F1 (0.5324, MCC 0.3710). Despite reasonable accuracy (0.7594), Precision and Recall are both low. KNN is sensitive to the high dimensionality introduced by one-hot encoding (30 features post-preprocessing) - distances become less meaningful in higher dimensions. AUC of 0.7847 shows the model has some discriminative power but struggles at the classification threshold. |
| Naive Bayes | Highest Recall (0.8663) but lowest Accuracy (0.6558). The Gaussian independence assumption is violated here - features like tenure, MonthlyCharges and TotalCharges are strongly correlated - causing overconfident predictions. However, the competitive AUC (0.8096) shows that the probability estimates are reasonably well-ordered. Useful when catching all churners matters more than precision. |
| Random Forest (Ensemble) | Best Precision (0.6267) and second-best AUC (0.8215). Bagging over 100 trees substantially corrects the Decision Tree's overfitting - AUC improves from 0.7587 to 0.8215. Lower Recall (0.4893) means it misses a notable share of actual churners, making it more conservative. Well-suited when false positives (incorrect churn alerts) are costly. |
| **Overall Winner** | **Logistic Regression** - best on 4 of 6 metrics (Accuracy, AUC, F1, MCC). The churn prediction problem has a predominantly linear structure after preprocessing, making Logistic Regression the most accurate, interpretable, and balanced choice for this dataset. |

---

## Live Streamlit App

**https://ml-classification-models.streamlit.app/**

---

## Setup and Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/j-v-sravan/ML-Classification-Models.git
cd ML-Classification-Models

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset.csv in the project root (download from Kaggle: Telco Customer Churn)

# 4. Train models and generate test_data.csv
python model/train_models.py

# 5. Launch Streamlit app
streamlit run app.py
```

---

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to https://streamlit.io/cloud and sign in with GitHub
3. Click New App and select your repository
4. Set branch to main and main file to app.py
5. Click Deploy

The .pkl model files and test_data.csv must be committed to the repo
so Streamlit Cloud can load them without re-training.
