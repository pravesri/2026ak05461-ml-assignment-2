# Iris Classification - ML Models Comparison

## a) Problem Statement

The Iris classification problem is a classic machine learning task that involves predicting the species of an iris flower based on its physical characteristics. Given measurements of sepal length, sepal width, petal length, and petal width, the goal is to classify the iris into one of three species: Setosa, Versicolor, or Virginica.

This is a **multi-class classification problem** where we implement and compare 5 different machine learning algorithms to determine which performs best on this dataset.

---

## b) Dataset Description

**Dataset Name:** Iris Dataset (Fisher's Iris Dataset)

**Source:** UCI Machine Learning Repository / scikit-learn

**Problem Type:** Multi-class Classification (3 classes)

**Number of Features:** 4
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

**Number of Instances:** 150 total (105 training, 45 testing - 70/30 split)

**Target Variable:** Iris Species
- Class 0: Iris Setosa (50 samples)
- Class 1: Iris Versicolor (50 samples)
- Class 2: Iris Virginica (50 samples)

**Data Characteristics:**
- Balanced dataset with equal distribution across classes
- Continuous numerical features
- No missing values
- Features are standardized using StandardScaler for better model performance

---

## c) GitHub Repository Link

**Repository:** [Insert Your GitHub Repository Link Here]

This repository contains:
- ✅ Complete source code (model_training.py, app.py)
- ✅ requirements.txt with all dependencies
- ✅ README.md with detailed documentation
- ✅ test_data.csv (test dataset with features and target)
- ✅ model/ folder with saved trained models (.joblib files)
- ✅ Streamlit app for interactive model evaluation

---

## d) Models Used: Performance Comparison

### Model Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| K-Nearest Neighbor | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Support Vector Machine | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## Model Observations & Performance Analysis

### Logistic Regression
**Observations:** Logistic Regression performs exceptionally well on the Iris dataset, achieving perfect classification on the test set (100% accuracy). The model benefits from the relatively simple, linearly separable nature of the Iris dataset. This linear classifier finds clear decision boundaries between the three iris species based on the four features.

**Strengths:**
- Fast training and inference time
- Highly interpretable coefficients
- Probabilistic predictions
- Excellent generalization

---

### Decision Tree
**Observations:** The Decision Tree classifier also achieves perfect accuracy on the test set. The Iris dataset's feature hierarchy allows the tree to learn clear splitting rules. The decision tree structure is interpretable, allowing us to understand exactly how the model makes decisions at each node.

**Strengths:**
- Perfect separation of classes
- Easy to visualize and interpret
- Handles multi-class problems naturally
- No feature scaling required

**Considerations:** May be prone to overfitting on more complex datasets.

---

### K-Nearest Neighbor
**Observations:** KNN demonstrates perfect classification performance on the Iris dataset. With k=5, the algorithm finds the 5 nearest neighbors for each sample and assigns the majority class. The Iris dataset's natural clustering makes KNN highly effective.

**Strengths:**
- Simple and intuitive algorithm
- No training phase required
- Perfect separation achievable with optimal k
- Effective for multi-class problems

**Considerations:** Computational cost increases with dataset size; sensitive to feature scaling (addressed with StandardScaler).

---

### Naive Bayes
**Observations:** The Gaussian Naive Bayes classifier achieves excellent performance (100% accuracy) despite making strong independence assumptions between features. The Iris dataset's structure allows the model to learn distinct probability distributions for each class.

**Strengths:**
- Fast and efficient
- Works well with high-dimensional data
- Probabilistic framework
- Requires minimal training data

**Considerations:** Independence assumption may not hold for correlated features, though it still performs well here.

---

### Support Vector Machine
**Observations:** SVM with RBF kernel achieves perfect classification on the test set. The non-linear kernel allows the model to find complex decision boundaries that perfectly separate the three iris species. The RBF kernel's ability to map the data to higher dimensions proves highly effective.

**Strengths:**
- Excellent at finding non-linear boundaries
- Memory efficient with support vectors
- Strong generalization
- Effective for multi-class problems

**Considerations:** Requires feature scaling (applied with StandardScaler) and hyperparameter tuning for different datasets.

---

## Overall Performance Summary

| Metric | Winner(s) | Performance |
|---|---|---|
| **Best Accuracy** | All Models (Tie) | 1.0000 (100%) |
| **Best Precision** | All Models (Tie) | 1.0000 (100%) |
| **Best Recall** | All Models (Tie) | 1.0000 (100%) |
| **Best F1 Score** | All Models (Tie) | 1.0000 (100%) |
| **Best MCC** | All Models (Tie) | 1.0000 (1.0) |

### **Overall Winner for the Iris Dataset: All Models (Perfect Tie)**

**Conclusion:** All five ML models achieve perfect classification on the Iris dataset. This outcome is not surprising because:

1. **Dataset Simplicity:** The Iris dataset is a relatively simple classification problem with clear, well-separated classes
2. **Feature Quality:** The four features (sepal/petal dimensions) provide strong discriminative power
3. **Sample Size:** With 150 samples and only 3 classes, all models have sufficient training data
4. **Linear Separability:** The classes are largely linearly separable, benefiting all algorithms

**Recommendation:** For the Iris dataset, **Logistic Regression** is recommended as the best choice due to:
- ✅ Fastest inference time
- ✅ Most interpretable (clear feature coefficients)
- ✅ Perfect accuracy with minimal complexity
- ✅ Lowest computational cost
- ✅ Best for production deployment

While all models perform equally well on test metrics, Logistic Regression's simplicity, interpretability, and speed make it the most practical choice for deployment.

---

## Project Structure

```
iris-classification/
│
├── app.py                    # Streamlit web application
├── model_training.py         # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── test_data.csv            # Test dataset (CSV format)
│
└── models/                  # Saved trained models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbor.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── scaler.joblib        # StandardScaler for feature scaling
    └── iris_dataset.joblib  # Dataset reference
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional - Pre-trained models included)
```bash
python model_training.py
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Streamlit App Features

1. **📈 Model Comparison Tab**
   - View all models' metrics in a comparison table
   - Visualize performance across all 6 metrics
   - Identify best performing model

2. **🧪 Test Predictions Tab**
   - Upload custom test CSV data
   - Select any model for prediction
   - View real-time evaluation metrics
   - Generate confusion matrix
   - Generate classification report

3. **📋 Model Details Tab**
   - Learn about each algorithm
   - View individual model performance
   - Understand strengths and weaknesses

---

## Evaluation Metrics Explained

- **Accuracy:** Proportion of correct predictions (TP+TN)/(Total)
- **AUC (Area Under ROC Curve):** Measures model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are correct (TP)/(TP+FP)
- **Recall:** Proportion of actual positives correctly identified (TP)/(TP+FN)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Correlation between predictions and actual, ranges -1 to 1

---

## Testing

Test data is provided in `test_data.csv` with 45 samples and their true labels. The Streamlit app allows:
- Uploading the test data
- Making predictions with each model
- Comparing performance metrics
- Visualizing confusion matrices

---

## Key Findings

1. **All models achieve 100% accuracy** on the Iris test set, indicating dataset simplicity
2. **Logistic Regression** is optimal for deployment: fast, interpretable, and perfect accuracy
3. **Feature scaling** is essential, especially for distance-based methods (KNN, SVM)
4. The Iris dataset demonstrates clear class separation in the feature space
5. All 6 evaluation metrics show perfect performance, indicating no overfitting

---

## Dependencies

- `streamlit` - Web application framework
- `scikit-learn` - ML algorithms and metrics
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Visualizations
- `joblib` - Model serialization

---

## Author
Praveen Srivastava  
M.Tech (AIML) - BITS Pilani

**Submission Date:** November 8, 2026

---

**License:** Educational Purpose  
**Dataset Source:** UCI Machine Learning Repository / scikit-learn
