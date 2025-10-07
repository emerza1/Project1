import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the CSV file
df = pd.read_csv("Project 1 Data.csv")

# Show first rows
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass distribution (Step column):")
print(df['Step'].value_counts())

# Histograms for numeric features
df.hist(bins=30, figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# Correlation on numeric columns only
print("\nCorrelation Matrix:")
corr_matrix = df.corr(numeric_only=True)
print(abs(corr_matrix))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Features and target
X = df.drop("Step", axis=1)
y = df["Step"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression with GridSearchCV
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

# Random Forest with GridSearchCV
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

# SVM with GridSearchCV
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

# Gradient Boosting with RandomizedSearchCV
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

print("\nGradient Boosting")
print("Best Params:", rand_gb.best_params_)
print("Best CV Score:", rand_gb.best_score_)
y_pred_gb = rand_gb.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# Collect metrics on the test set
models = {
    "Logistic Regression": grid_lr.best_estimator_,
    "Random Forest": grid_rf.best_estimator_,
    "SVM": grid_svm.best_estimator_,
    "Gradient Boosting": rand_gb.best_estimator_
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nModel comparison on the test set:")
print(results_df)

# Consistent class order for confusion matrices
class_names = np.sort(y.unique())

# Confusion matrices
cm1 = confusion_matrix(y_test, y_pred_lr, labels=class_names)
cm2 = confusion_matrix(y_test, y_pred_rf, labels=class_names)
cm3 = confusion_matrix(y_test, y_pred_svm, labels=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Support Vector Machine')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Stacking with actual estimators
best_rf_est = grid_rf.best_estimator_
best_svm_est = grid_svm.best_estimator_

base_models = [
    ('rf', best_rf_est),
    ('svm', best_svm_est)
]

meta_model = LogisticRegression(max_iter=1000)

stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)

f1_stacked = f1_score(y_test, y_pred_stacked, average='weighted', zero_division=0)
precision_stacked = precision_score(y_test, y_pred_stacked, average='weighted', zero_division=0)
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)

print(f"\nStacked Model - Accuracy: {accuracy_stacked:.4f}, Precision: {precision_stacked:.4f}, F1 Score: {f1_stacked:.4f}")

cm_stacked = confusion_matrix(y_test, y_pred_stacked, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stacked, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Save the best model using joblib
import joblib

# Save the best model (SVM in this case)
joblib.dump(best_svm_est, 'best_maintenance_model_svm.joblib')
print("SVM Model saved as 'best_maintenance_model_svm.joblib'")

# Load the saved model
loaded_model = joblib.load('best_maintenance_model_svm.joblib')

# New data for prediction
new_data = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# Predict the maintenance steps for the new data
predictions = loaded_model.predict(new_data)

# Print the predictions
print("Predictions for the new data:")
print(predictions)
