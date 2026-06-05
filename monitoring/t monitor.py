import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
import os

# ─────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────
print("📂 Loading data...")

reference_data = pd.read_csv('data/processed/churn_features.csv')

np.random.seed(42)
production_data = reference_data.copy()

# Simulate drift
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
# Step 3: Create Report
# ─────────────────────────────────────
print("\n📊 Generating drift report...")

report = Report(metrics=[
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnDriftMetric(column_name='MonthlyCharges'),
    ColumnDriftMetric(column_name='tenure'),
    ColumnDriftMetric(column_name='TotalCharges'),
])

report.run(
    reference_data=reference_data,
    current_data=production_data
)

# ─────────────────────────────────────
# Step 4: Save Report
# ─────────────────────────────────────
report.save_html('monitoring/drift_report.html')
print("✅ Drift report saved!")

print("\n🎉 Monitoring Complete!")