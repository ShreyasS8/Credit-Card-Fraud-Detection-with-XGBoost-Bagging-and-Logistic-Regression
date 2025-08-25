# Credit Card Fraud Detection (Kaggle Dataset)

This repository implements multiple machine learning models for **credit
card fraud detection** using the [Kaggle Credit Card Fraud
dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 📂 Dataset

-   **Source:**
    [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)\
-   **Size:** 284,807 transactions\
-   **Features:** 30 (28 PCA-anonymized features, plus `Time` and
    `Amount`)\
-   **Target:** `Class` (0 = legitimate, 1 = fraud)\
-   **Fraud rate:** \~0.17% (492 frauds out of 284,807)

To run the code, place the dataset file at:

    /kaggle/input/creditcardfraud/creditcard.csv

## ⚙️ Models Implemented

1.  **Logistic Regression** (baseline, class-weight balanced)\
2.  **Bagging Classifier** (Decision Tree ensemble)\
3.  **XGBoost** (with and without SMOTE oversampling)\
4.  **RandomizedSearchCV** for hyperparameter tuning (XGBoost &
    SMOTE-XGBoost)

Additional features: - Automatic **threshold selection** (best F1 from
PR curve)\
- **Precision@K** and **Lift@K** metrics\
- **Confusion matrix analysis** (counts and percentages)\
- **Permutation importance** for feature ranking\
- Optional **SHAP explainability** plots

## 🧪 Experimental Setup

-   **Train/Test split:** 80/20, stratified\
-   **Scaling:** StandardScaler applied to `Time` and `Amount`\
-   **Random seed:** 42\
-   **Evaluation metrics:**
    -   AUPRC (Average Precision / PR-AUC)\
    -   ROC AUC\
    -   Precision, Recall, F1\
    -   Precision@100 and Lift@100

## 📊 Results (Test Set)

  -----------------------------------------------------------------------------------------------------------------------
  Model                 AUPRC    ROC_AUC   BestThresh   BestF1   Precision   Recall   F1       Precision@100   Lift@100
  --------------------- -------- --------- ------------ -------- ----------- -------- -------- --------------- ----------
  XGBoost_noSMOTE       0.8663   0.9760    0.6663       0.8646   0.8737      0.8469   0.8601   0.8300          482.43

  Bagging               0.8605   0.9533    0.4900       0.8877   0.9222      0.8469   0.8830   0.8500          494.05

  Logistic Regression   0.7189   0.9721    1.0000       0.8247   0.8247      0.8163   0.8205   0.8000          464.99

  SMOTE_XGBoost_tuned   0.7015   0.9620    0.9069       0.7876   0.7600      0.7755   0.7677   0.7600          441.74
  -----------------------------------------------------------------------------------------------------------------------

➡️ **Best Models:**\
- **Bagging** achieved the highest F1 (0.8830) with strong precision
(0.9222).\
- **XGBoost_noSMOTE** had the best AUPRC (0.8663).

## 📦 Requirements

Install dependencies before running:

``` bash
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn shap seaborn joblib
```

## ▶️ Usage

Run the training and evaluation pipeline:

``` bash
python fraud_detection.py
```

Outputs are saved in `models_outputs/`: - Trained models (`.joblib`)\
- Evaluation summary (`evaluation_summary.csv`)\
- Confusion matrices (`*_confusion_matrix.csv`)\
- Skipped steps (`skipped_steps.txt`, if any)

## 📌 Notes

-   SHAP plots are optional (`pip install shap`)\
-   SMOTE requires `imbalanced-learn`\
-   Designed for Kaggle/Colab but works locally with correct dataset
    path

------------------------------------------------------------------------

**Author:** Shreyas Shimpi
