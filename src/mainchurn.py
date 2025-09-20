import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# See the first 5 rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.info())