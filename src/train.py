import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import os

# ─────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv('data/processed/churn_features.csv')
print("✅ Data loaded:", df.shape)

# ─────────────────────────────────────
# Step 2: Split Features & Target
# ─────────────────────────────────────
X = df.drop('Churn', axis=1)   # everything except Churn
y = df['Churn']                 # only Churn column

print("✅ Features shape:", X.shape)
print("✅ Target shape:", y.shape)

# ─────────────────────────────────────
# Step 3: Split Train & Test
# ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # same split every time
    stratify=y          # keep churn ratio same
)

print("✅ Training samples:", X_train.shape[0])
print("✅ Testing samples:", X_test.shape[0])

# ─────────────────────────────────────
# Step 4: Train Model with MLflow
# ─────────────────────────────────────
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():

    # Define model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }

    # Log parameters to MLflow
    mlflow.log_params(params)

    # Train XGBoost model
    print("\n🤖 Training XGBoost model...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("✅ Model trained!")

    # ─────────────────────────────────────
    # Step 5: Evaluate Model
    # ─────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Print metrics
    print("\n📊 Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

    # ─────────────────────────────────────
    # Step 6: Save Model
    # ─────────────────────────────────────
    mlflow.xgboost.log_model(model, "model")
    model.save_model('models/churn_model.json')
    print("\n✅ Model saved to models/churn_model.json")
    print("✅ Model logged to MLflow!")

print("\n🎉 Training Complete!")