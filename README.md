# Banking Fraud Detection System

## Project Overview
This project implements a machine learning pipeline to detect fraudulent banking transactions using a Random Forest Classifier. It addresses data imbalance with SMOTE and applies normalization for optimal model performance. The system provides evaluation metrics and visualizations to understand the model's effectiveness and feature importance.

## Features
- Data loading and cleaning (handling missing values)
- Encoding of categorical variables
- Handling class imbalance using SMOTE
- Data normalization using StandardScaler
- Random Forest model training and prediction
- Evaluation via confusion matrix, classification report, and ROC AUC score
- Visualization of class distribution and feature importance

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

Install required packages using:



## Usage
1. Place your dataset file named `test.CSV` in the project directory.
2. Run the Python script to perform fraud detection.
3. Visualizations will display class distribution and feature importance.
4. The console will output model evaluation metrics.

## Dataset
The dataset should include:
- `TransactionID` (unique transaction identifier)
- Features describing transactions (numerical and categorical)
- `Class` column (0 for non-fraud, 1 for fraud)

## How It Works
- Cleans data by dropping missing values.
- Converts categorical variables using one-hot encoding.
- Balances the data with SMOTE.
- Splits data into training and testing subsets.
- Normalizes features for better model performance.
- Trains a Random Forest classifier.
- Provides evaluation metrics and visual insights.

## Evaluation Metrics
- **Confusion Matrix:** Breakdown of prediction results.
- **Classification Report:** Precision, recall, F1-score for each class.
- **ROC AUC Score:** Overall metric for model discrimination ability.
- **Feature Importance:** Shows influence of each feature on predictions.

## Author
[Your Name]

## License
Specify your license here (e.g., MIT License).
