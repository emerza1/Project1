import pandas as pd

# Read the CSV file
df = pd.read_csv("Project 1 Data.csv")

# Print first few rows
print(df.head())
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Basic Statistical Analysis
# -----------------------------

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass distribution (Step column):")
print(df['Step'].value_counts())

# -----------------------------
# 2. Visualization
# -----------------------------

# Histogram of each numeric feature
df.hist(bins=30, figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()
