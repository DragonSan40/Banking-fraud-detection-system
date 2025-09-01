# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('test.CSV')

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, palette='Set2')
plt.title('Class Distribution (Fraud vs Non-Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected, dropping rows with missing values.")
    data = data.dropna()  # Remove rows with missing values

# Features and target
X = data.drop(['Class', 'TransactionID'], axis=1)  # Exclude 'Class' and 'TransactionID'
y = data['Class']  # Target column (0 = Non-fraud, 1 = Fraud)

# Handle categorical features by encoding them
X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for categorical variables

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Evaluation Results:")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importances = model.feature_importances_
feature_names = X.columns
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title('Feature Importances from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
