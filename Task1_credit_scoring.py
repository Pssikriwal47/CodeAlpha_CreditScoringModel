"""
CodeAlpha ML Task 1: Credit Scoring Model
Predicts creditworthiness (Good=1 / Bad=0) from financial history features.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, classification_report, confusion_matrix)

# ---------------------------------------------------------------
# STEP 2: Generate a realistic synthetic credit dataset
# (In a real submission you can swap this for a Kaggle "Give Me Some
# Credit" / German Credit dataset CSV with the same column names.)
# ---------------------------------------------------------------
np.random.seed(42)
n = 3000

income = np.random.normal(55000, 20000, n).clip(8000, 200000)
age = np.random.randint(21, 65, n)
debts = np.random.normal(15000, 10000, n).clip(0, 100000)
num_late_payments = np.random.poisson(1.2, n)
credit_history_years = np.random.randint(0, 30, n)
num_open_accounts = np.random.randint(1, 15, n)
utilization_ratio = np.random.beta(2, 5, n)  # credit used / credit limit

# Debt-to-income ratio (key engineered feature)
debt_to_income = debts / income

# Latent "creditworthiness" score combining the features (ground truth rule)
risk_score = (
    -0.00002 * income
    + 3.5 * debt_to_income
    + 0.6 * num_late_payments
    - 0.04 * credit_history_years
    + 2.0 * utilization_ratio
    - 0.02 * (age - 21)
    + np.random.normal(0, 0.8, n)  # noise
)
# Lower risk_score -> good credit (1), higher -> bad credit (0)
threshold = np.percentile(risk_score, 70)
creditworthy = (risk_score < threshold).astype(int)

df = pd.DataFrame({
    "income": income,
    "age": age,
    "debts": debts,
    "num_late_payments": num_late_payments,
    "credit_history_years": credit_history_years,
    "num_open_accounts": num_open_accounts,
    "utilization_ratio": utilization_ratio,
    "debt_to_income": debt_to_income,
    "creditworthy": creditworthy
})
df.to_csv("/home/claude/task1_credit_scoring/credit_data.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(df["creditworthy"].value_counts(normalize=True).rename("class_ratio"))

# ---------------------------------------------------------------
# STEP 3: Train/test split + scaling
# ---------------------------------------------------------------
X = df.drop(columns=["creditworthy"])
y = df["creditworthy"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# STEP 4: Train multiple classification algorithms
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1]

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    results.append({
        "Model": name, "Precision": round(precision, 3),
        "Recall": round(recall, 3), "F1-Score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3)
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["Bad", "Good"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# ---------------------------------------------------------------
# STEP 5: Compare models
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))
results_df.to_csv("/home/claude/task1_credit_scoring/model_comparison.csv", index=False)

# Feature importance from Random Forest (best interpretability for report)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Feature Importance (Random Forest) ===")
print(importances)
importances.to_csv("/home/claude/task1_credit_scoring/feature_importance.csv")
