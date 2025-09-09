import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(df):
    """
    Preprocesses the telecom churn dataset.
    - Handles missing values
    - Encodes categorical features
    """
    # Drop customerID as it's not a useful feature for prediction
    df = df.drop('customerID', axis=1)

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Impute missing 'TotalCharges' with the median
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)

    # Encode categorical variables using LabelEncoder
    # This converts categorical text data into a numerical format
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Churn': # We will encode the target variable separately
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Encode the target variable 'Churn'
    le_churn = LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'])
    
    print("Data Preprocessing Complete.")
    print("Sample of preprocessed data:")
    print(df.head())
    
    return df, le_churn

def train_and_evaluate_model(df):
    """
    Trains a RandomForestClassifier and evaluates its performance.
    """
    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split the data into training and testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nData split into training and testing sets.")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Initialize the RandomForestClassifier model
    # n_estimators is the number of trees in the forest
    # random_state ensures reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

    # Train the model
    print("\nTraining the Random Forest model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Display classification report
    print("\nClassification Report:")
    # target_names uses the label encoder to show 'No' and 'Yes' instead of 0 and 1
    print(classification_report(y_test, y_pred, target_names=le_churn.classes_))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_churn.classes_, yticklabels=le_churn.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Churn')
    plt.xlabel('Predicted Churn')
    plt.show()

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 8))
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Important Features')
    plt.xlabel('Feature Importance Score')
    plt.show()

if __name__ == '__main__':
    # Load the dataset
    try:
        df = pd.read_csv('telecom_churn.csv')
        print("Dataset 'telecom_churn.csv' loaded successfully.")
        
        # Preprocess the data
        processed_df, le_churn = preprocess_data(df)
        
        # Train and evaluate the model
        train_and_evaluate_model(processed_df)
        
    except FileNotFoundError:
        print("Error: 'telecom_churn.csv' not found.")
        print("Please make sure the CSV file is in the same directory as the script.")