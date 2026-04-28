import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load cleaned data
df = pd.read_csv('data/processed/churn_cleaned.csv')
print("✅ Data loaded:", df.shape)

# Step 1: Remove customerID (not useful for prediction)
df = df.drop('customerID', axis=1)
print("✅ Removed customerID column")

# Step 2: Convert Yes/No columns to 1/0
binary_columns = ['Partner', 'Dependents', 'PhoneService',
                  'PaperlessBilling', 'Churn']

for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
print("✅ Converted Yes/No columns to 1/0")

# Step 3: Convert gender to 1/0
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
print("✅ Converted gender to 1/0")

# Step 4: Label encode remaining text columns
text_columns = ['MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod']

le = LabelEncoder()
for col in text_columns:
    df[col] = le.fit_transform(df[col])
print("✅ Encoded all text columns")

# Step 5: Scale numerical columns
scaler = StandardScaler()
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print("✅ Scaled numerical columns")

# Save processed data
df.to_csv('data/processed/churn_features.csv', index=False)
print("✅ Feature engineering complete!")
print("Final shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())