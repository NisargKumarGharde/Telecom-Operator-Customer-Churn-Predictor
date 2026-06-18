# Telecom Operator Customer Churn Predictor

<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" height="48" alt="Python" title="Python" /> &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" height="48" alt="Pandas" title="Pandas" /> &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" height="48" alt="Scikit-Learn" title="Scikit-Learn" /> &nbsp;
  <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" height="48" alt="Seaborn" title="Seaborn" /> &nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" height="48" alt="Matplotlib" title="Matplotlib" />
</p>

## Project Overview
This project provides a machine learning pipeline designed to predict customer churn for telecom operators. By analyzing customer demographics, service usage, and billing data, the model identifies at-risk customers, allowing businesses to optimize their retention strategies proactively.

## Features
- **Data Preprocessing Pipeline:** Automated handling of missing values (median imputation) and categorical encoding using `LabelEncoder`.
- **Predictive Modeling:** Implements a `RandomForestClassifier` ensemble model for robust churn prediction.
- **Evaluation Metrics:** Generates comprehensive performance reports including Accuracy, Precision, Recall, and Confusion Matrices.
- **Feature Importance Analysis:** Extracts and visualizes the top drivers of customer churn (e.g., Contract Type, Monthly Charges) to provide actionable business insights.

## Why I Built This

[#why-i-built-this](#why-i-built-this)

This project is the classical-ML counterpart to my [GCP Customer Churn Intelligence Platform](https://github.com/NisargKumarGharde/gcp-churn-intelligence-platform.git): same problem domain, deliberately different execution. Where the GCP project focuses on production deployment (streaming ingestion, serverless model serving, cloud-native ELT), this one is about getting the modeling fundamentals right end-to-end: clean preprocessing, a well-evaluated baseline model, and clear feature importance analysis, the kind of pipeline you'd build before deciding anything is worth productionizing.

## Engineering Decisions

[#engineering-decisions](#engineering-decisions)

**Why RandomForest?** It handles mixed categorical/numerical features well without heavy preprocessing, and its built-in feature importance gives directly actionable business insight (which is the actual point of churn modeling) rather than just a black-box prediction.

**Why median imputation?** Telecom billing and usage fields are right-skewed (a few customers with very high charges/tenure), so median is more robust to outliers than mean imputation.

## What's Next

[#whats-next](#whats-next)

- Compare against XGBoost/LightGBM baselines
- Add cross-validation and hyperparameter tuning instead of a single train/test split
- Wrap the trained model behind a small Flask/FastAPI endpoint for inference

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
