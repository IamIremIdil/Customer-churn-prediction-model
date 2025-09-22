import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download the CORRECT Telco Customer Churn dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Dataset downloaded to:", path)

# List what files were downloaded
import os
print("Files in downloaded folder:")
for file in os.listdir(path):
    print(" -", file)

# Load the data (adjust filename if needed)
df = pd.read_csv(path + '/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Continue with your analysis...
print(df.head())
print(df.isnull().sum())

# Basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Number of customers: {len(df)}")

# Check the distribution of the target variable (Churn)
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))  # Percentages

# Start some basic visualizations
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Churn')
plt.title('Customer Churn Distribution')
plt.show()

# Check numerical columns statistics
print("\nNumerical columns summary:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

# Check categorical columns
print("\nCategorical columns unique values:")
categorical_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")