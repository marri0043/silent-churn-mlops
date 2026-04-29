import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os

# ─────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv('data/processed/churn_features.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']
print("✅ Data loaded:", X.shape)

# ─────────────────────────────────────
# Step 2: Load Trained Model
# ─────────────────────────────────────
print("\n📂 Loading trained model...")
model = XGBClassifier()
model.load_model('models/churn_model.json')
print("✅ Model loaded!")

# ─────────────────────────────────────
# Step 3: Create SHAP Explainer
# ─────────────────────────────────────
print("\n🔍 Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("✅ SHAP values calculated!")

# ─────────────────────────────────────
# Step 4: Plot Feature Importance
# ─────────────────────────────────────
print("\n📊 Creating SHAP plots...")

# Plot 1: Summary plot - most important features
plt.figure()
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)
plt.title("Top Features Causing Churn")
plt.tight_layout()
plt.savefig('monitoring/shap_feature_importance.png')
plt.close()
print("✅ Feature importance plot saved!")

# Plot 2: Detailed summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    X,
    show=False
)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig('monitoring/shap_summary.png')
plt.close()
print("✅ Summary plot saved!")

# ─────────────────────────────────────
# Step 5: Explain Single Customer
# ─────────────────────────────────────
print("\n👤 Explaining single customer prediction...")

# Take first customer
customer = X.iloc[0:1]
customer_shap = explainer.shap_values(customer)

print(f"\nCustomer Features:")
print(customer.T)

print(f"\nSHAP Values (impact on prediction):")
feature_impact = pd.DataFrame({
    'Feature': X.columns,
    'SHAP Value': customer_shap[0]
}).sort_values('SHAP Value', ascending=False)

print(feature_impact)
print("\n✅ Positive SHAP = pushes towards CHURN")
print("✅ Negative SHAP = pushes towards STAYING")

print("\n🎉 Explanation Complete!")