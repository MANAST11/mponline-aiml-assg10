import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("=== Task 1: Data Understanding and Preprocessing ===")
    
    # 1. Load the dataset using Pandas.
    df = pd.read_csv('heart.csv')
    
    # 2. Display the first five records.
    print("\nFirst five records of the dataset:")
    print(df.head())
    
    # 3. Identify Numerical features and Target variable.
    # In this dataset, all input columns are numerical (either continuous or integer-encoded categories).
    target_col = 'target'
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"\nTarget Variable: {target_col}")
    print(f"Numerical Features (Inputs): {feature_cols}")
    
    # 4. Check for missing values.
    missing_vals = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_vals)
    
    # 5. Split the dataset into 80% training and 20% testing.
    X = df[feature_cols]
    y = df[target_col]
    
    # Using random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nDataset Split Successful:")
    print(f"Training Set Shape: {X_train.shape}")
    print(f"Testing Set Shape: {X_test.shape}")
    
    print("\n=== Task 2: Model Development ===")
    
    # Build classification model (Random Forest Classifier)
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    # Save the trained model using Joblib
    model_filename = 'model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel saved successfully as '{model_filename}' using joblib!")

if __name__ == '__main__':
    main()
