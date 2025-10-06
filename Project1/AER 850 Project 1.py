import pandas as pd

# Read the CSV file
df = pd.read_csv("Project 1 Data.csv")

# Print first few rows
print(df.head())
import numpy as np
import matplotlib.pyplot as plt

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass distribution (Step column):")
print(df['Step'].value_counts())

# Histogram of each numeric feature
df.hist(bins=30, figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

import seaborn as sns

# Correlation matrix
print("\nCorrelation Matrix:")
corr_matrix = df.corr()
print(abs(corr_matrix))

sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Features and target (assuming 'Step' is the target)
X = df.drop("Step", axis=1)
y = df["Step"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression (GridSearchCV)
param_grid_lr = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_lr.fit(X_train, y_train)

print("\nLogistic Regression")
print("Best Params:", grid_lr.best_params_)
print("Best CV Score:", grid_lr.best_score_)
y_pred_lr = grid_lr.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# Random Forest (GridSearchCV)
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

print("\nRandom Forest")
print("Best Params:", grid_rf.best_params_)
print("Best CV Score:", grid_rf.best_score_)
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Support Vector Machine (GridSearchCV)
param_grid_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
grid_svm = GridSearchCV(
    SVC(),
    param_grid_svm,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_svm.fit(X_train, y_train)

print("\nSupport Vector Machine")
print("Best Params:", grid_svm.best_params_)
print("Best CV Score:", grid_svm.best_score_)
y_pred_svm = grid_svm.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Gradient Boosting (RandomizedSearchCV)
param_dist_gb = {
    "n_estimators": np.arange(50, 301, 50),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9]
}
rand_gb = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist_gb,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
rand_gb.fit(X_train, y_train)

print("\nGradient Boosting (RandomizedSearchCV)")
print("Best Params:", rand_gb.best_params_)
print("Best CV Score:", rand_gb.best_score_)
y_pred_gb = rand_gb.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
