"""
Model Training Script for Iris Classification
Trains 5 ML models and calculates performance metrics
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)
import joblib
import os

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data as CSV
test_df = pd.DataFrame(X_test_scaled, columns=iris.feature_names)
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("✓ Test data saved as test_data.csv")

# Create models directory
os.makedirs('models', exist_ok=True)

# Define all 5 ML models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
}

# Train models and collect metrics
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Calculate all 6 metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })

    # Save trained model
    model_filename = model_name.replace(" ", "_").lower().replace("-", "_")
    joblib.dump(model, f'models/{model_filename}.joblib')
    print(f"  ✓ Model saved: models/{model_filename}.joblib")

# Create results dataframe and save
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)

print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Save scaler for use in Streamlit app
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(iris, 'models/iris_dataset.joblib')

print("\n✓ All models trained and saved!")
print("✓ Results saved to model_results.csv")
print("✓ Scaler saved for deployment")
