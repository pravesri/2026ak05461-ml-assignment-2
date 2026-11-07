"""
Streamlit Web Application for Iris Classification Models
Interactive ML Model Deployment and Evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Iris Classification Models",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌸 Iris Classification - ML Models Comparison")
st.markdown("""
This application demonstrates **5 different machine learning classification models**
trained on the Iris dataset. Upload test data to see predictions and evaluation metrics.
""")

# Sidebar
st.sidebar.header("📊 Configuration")

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.joblib',
        'Decision Tree': 'models/decision_tree.joblib',
        'K-Nearest Neighbor': 'models/k_nearest_neighbor.joblib',
        'Naive Bayes': 'models/naive_bayes.joblib',
        'Support Vector Machine': 'models/support_vector_machine.joblib'
    }

    scaler = joblib.load('models/scaler.joblib')

    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            models[model_name] = joblib.load(file_path)

    return models, scaler

try:
    models, scaler = load_models()
    st.sidebar.success(f"✓ Loaded {len(models)} models")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load pre-computed results
@st.cache_data
def load_results():
    return pd.read_csv('model_results.csv')

results_df = load_results()

# Tab 1: Model Comparison
tab1, tab2, tab3 = st.tabs(["📈 Model Comparison", "🧪 Test Predictions", "📋 Model Details"])

with tab1:
    st.subheader("Model Performance Metrics Comparison")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    with col2:
        st.metric("Best Model (Accuracy)",
                 results_df.loc[results_df['Accuracy'].idxmax(), 'Model'],
                 f"{results_df['Accuracy'].max():.4f}")

    # Visualization
    st.subheader("Metrics Visualization")

    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        axes[idx].barh(results_df['Model'], results_df[metric], color='steelblue')
        axes[idx].set_xlabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_xlim(0, 1)

        for i, v in enumerate(results_df[metric]):
            axes[idx].text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Test Model Predictions")

    # File upload
    uploaded_file = st.file_uploader("Upload test data CSV (with features + target column)",
                                     type="csv")

    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)

        if 'target' not in test_data.columns:
            st.error("❌ CSV must contain a 'target' column")
        else:
            # Separate features and target
            X_test = test_data.drop('target', axis=1).values
            y_test = test_data['target'].values

            # Scale features
            X_test_scaled = scaler.transform(X_test)

            # Select model
            selected_model = st.selectbox("Select Model for Prediction:", list(models.keys()))

            # Get predictions
            model = models[selected_model]
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            try:
                y_pred_proba = model.predict_proba(X_test_scaled)
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = 0.0

            mcc = matthews_corrcoef(y_test, y_pred)

            # Display metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("AUC", f"{auc:.4f}")
            with col3:
                st.metric("Precision", f"{precision:.4f}")
            with col4:
                st.metric("Recall", f"{recall:.4f}")
            with col5:
                st.metric("F1 Score", f"{f1:.4f}")
            with col6:
                st.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                       yticklabels=['Setosa', 'Versicolor', 'Virginica'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred,
                                          target_names=['Setosa', 'Versicolor', 'Virginica'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

    else:
        st.info("📁 Upload a CSV file to make predictions")
        st.markdown("**Expected CSV format:**")
        st.code("sepal_length,sepal_width,petal_length,petal_width,target\n5.1,3.5,1.4,0.2,0")

with tab3:
    st.subheader("Model Descriptions")

    model_descriptions = {
        'Logistic Regression': """
        **Logistic Regression** is a linear classification algorithm that models
        the probability of a binary or multi-class outcome. It's interpretable and fast.
        """,

        'Decision Tree': """
        **Decision Tree** uses a tree-like model of decisions. It's easy to understand
        but prone to overfitting without pruning.
        """,

        'K-Nearest Neighbor': """
        **K-Nearest Neighbor (KNN)** classifies instances based on the majority class
        of k nearest neighbors. Simple but can be slow on large datasets.
        """,

        'Naive Bayes': """
        **Naive Bayes** is a probabilistic classifier based on Bayes' theorem.
        It assumes feature independence and is fast to train.
        """,

        'Support Vector Machine': """
        **Support Vector Machine (SVM)** finds the optimal hyperplane to maximize
        margin between classes. Effective for non-linear problems with kernel functions.
        """
    }

    for model_name, description in model_descriptions.items():
        with st.expander(f"ℹ️ {model_name}"):
            st.markdown(description)
            model_metrics = results_df[results_df['Model'] == model_name].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
            with col2:
                st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
            with col3:
                st.metric("MCC", f"{model_metrics['MCC']:.4f}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>🎓 ML Assignment 2 - Iris Classification Models Deployment</p>
    <p>Built with Streamlit | scikit-learn | Python</p>
</div>
""", unsafe_allow_html=True)
