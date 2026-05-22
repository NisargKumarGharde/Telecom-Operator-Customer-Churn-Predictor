# Telecom Operator Customer Churn Predictor

<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" height="48" alt="Python" title="Python" /> &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" height="48" alt="Scikit-Learn" title="Scikit-Learn" /> &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" height="48" alt="Pandas" title="Pandas" />
</p>

## Project Overview
This project provides a machine learning pipeline designed to predict customer churn for telecom operators. By analyzing customer demographics, service usage, and billing data, the model identifies at-risk customers, allowing businesses to optimize their retention strategies proactively.

## Features
- **Data Preprocessing Pipeline:** Automated handling of missing values (median imputation) and categorical encoding using `LabelEncoder`.
- **Predictive Modeling:** Implements a `RandomForestClassifier` ensemble model for robust churn prediction.
- **Evaluation Metrics:** Generates comprehensive performance reports including Accuracy, Precision, Recall, and Confusion Matrices.
- **Feature Importance Analysis:** Extracts and visualizes the top drivers of customer churn (e.g., Contract Type, Monthly Charges) to provide actionable business insights.

## Dataset
The model is trained on a telecom customer dataset containing:
- **Demographics:** Gender, Senior Citizen status, Partner/Dependents.
- **Account Information:** Tenure, Contract type, Payment method, Paperless billing.
- **Services Subscribed:** Phone, Multiple Lines, Internet, Online Security, Tech Support, Streaming.

## Quick Start

1. Ensure the dataset (`telecom_churn.csv`) is in the root directory.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Run the training and evaluation script:
   ```bash
   python "Telecom Operator Customer Churn Predictor.py"
   ```
