import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *
import os

# ─────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────
print("📂 Loading data...")

# Reference data = what model was trained on
reference_data = pd.read_csv('data/processed/churn_features.csv')

# Simulate production data with slight drift
# In real world this would be new incoming customer data
np.random.seed(42)
production_data = reference_data.copy()

# Simulate drift by changing some values
production_data['MonthlyCharges'] = production_data['MonthlyCharges'] + np.random.normal(0.5, 0.2, len(production_data))
production_data['tenure'] = production_data['tenure'] + np.random.normal(0.3, 0.1, len(production_data))

print("✅ Reference data shape:", reference_data.shape)
print("✅ Production data shape:", production_data.shape)

# ─────────────────────────────────────
# Step 2: Remove target column
# ─────────────────────────────────────
reference_data = reference_data.drop('Churn', axis=1)
production_data = production_data.drop('Churn', axis=1)

# ─────────────────────────────────────
# Step 3: Create Data Drift Report
# ─────────────────────────────────────
print("\n📊 Generating drift report...")

report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(
    reference_data=reference_data,
    current_data=production_data
)

# ─────────────────────────────────────
# Step 4: Save Report
# ─────────────────────────────────────
report.save_html('monitoring/drift_report.html')
print("✅ Drift report saved to monitoring/drift_report.html")

# ─────────────────────────────────────
# Step 5: Print Summary
# ─────────────────────────────────────
print("\n📊 Monitoring Summary:")
print("─────────────────────")
print("Reference Data: Training data (what model learned from)")
print("Production Data: New incoming customer data")
print("\n✅ Check monitoring/drift_report.html for full report!")
print("\n🎉 Monitoring Complete!")