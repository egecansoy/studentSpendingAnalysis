#region libraries
import pandas as pd
import numpy as np
#endregion

#region First Settings
pd.set_option('display.width', 1000)          # satır genişliği artır
pd.set_option('display.max_columns', None)   # tüm sütunlar
pd.set_option('display.expand_frame_repr', False)  # aşağı kaymayı engeller
#endregion

#region Data Load
df = pd.read_csv("C:/Users/alioz/PycharmProjects/studentSpendingAnalysis/data/raw/student_spending (1).csv")
#endregion

#region First Look
df.head(10)
print("Shape (satır,sütun)", df.shape)
print("\nColumns")
print(df.columns)
print("\nFirst 5 rows:")
df = df.drop("Unnamed: 0", axis=1)
print(df.head())
print("\nData Types:")
print(df.dtypes)
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Columns:")

# Her sütundaki benzersiz değer sayısı
unique_counts = df.nunique().sort_values()
print(unique_counts)
threshold = 10

cat_cols = [col for col in df.columns if df[col].nunique() < threshold]
num_cols = [col for col in df.columns if df[col].nunique() >= threshold]

print("Categorical columns:")
print(cat_cols)

print("\nNumerical columns:")
print(num_cols)

import seaborn as sns
import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")

plt.savefig("corr.png")
plt.close()
#endregion

#region Outlier Analysis
plt.figure(figsize=(6,4))
sns.boxplot(y=df["monthly_income"])
plt.title("Monthly Income Boxplot")
plt.show()

Q1 = df["monthly_income"].quantile(0.25)
Q3 = df["monthly_income"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["monthly_income"] < lower) | (df["monthly_income"] > upper)]

print("Lower:", lower)
print("Upper:", upper)
print("Outlier count:", outliers.shape[0])

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]

    print(f"{col} → Outlier: {outlier_count}")

#endregion
