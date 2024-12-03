# A Machine Learning Approach for Detection of Fraudulent Transactions in Financial Services

This project focuses on identifying fraudulent transactions in financial services using machine learning. We evaluated multiple models, addressed class imbalance, and analyzed feature importance using SHAP values. The project also includes hybrid approaches for improved performance.

## Problem Statement

Fraudulent transactions in financial services pose a significant challenge. This project aims to develop models capable of predicting fraud efficiently, thereby minimizing financial losses and improving customer trust.

## Objectives

- Detect fraudulent transactions using transaction-level features.
- Address class imbalance with under-sampling.
- Evaluate multiple machine learning models using various metrics.
- Explore hybrid models and feature importance using SHAP values.

## Dataset

### Source
- **Platform**: [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Files Used**: `train_transaction.csv` and `train_identity.csv`.

### Key Features
| **Feature Name**       | **Description**                          | **Type**         |
|-------------------------|------------------------------------------|------------------|
| TransactionDT           | Time of transaction                     | Continuous       |
| TransactionAmt          | Transaction amount                      | Continuous       |
| card1-card6             | Card details                            | Categorical      |
| addr1, addr2            | Billing addresses                       | Categorical      |
| isFraud                 | Target variable indicating fraud         | Categorical      |

### Data Preprocessing
- Balanced dataset created by under-sampling non-fraudulent transactions.
- Engineered features like `TransactionHour` and `Amt_to_mean_ratio`.
- Missing values handled and categorical features encoded.

This pie chart illustrates the proportion of fraudulent versus non-fraudulent transactions in the dataset.
<img src="/assets/images/Fraud vs Non Fraud Transactions.png" alt="Fraud vs Non-Fraud Transactions" width="500">

## Workflow

### Data Sampling and Feature Engineering
- **Under-sampling** balanced the dataset by reducing the non-fraudulent samples.
- **Feature Engineering** derived features such as TransactionHour, TransactionAmt_mean_card, and Amt_to_mean_ratio to enhance predictive power.

This histogram shows the distribution of transaction amounts before outlier removal.
<img src="/assets/images/Transaction Amt before outlier removal.png" alt="Transaction Amt before Outlier Removal" width="500">

This histogram shows the distribution of transaction amounts after outlier removal.
<img src="/assets/images/Transaction Amt after outlier removal.png" alt="Transaction Amt after Outlier Removal" width="500">


### Model Training and Evaluation
1. **Random Forest**
2. **Isolation Forest**
3. **Hybrid Models (RF + IF, RF + XGB)**
4. **XGBoost**
5. **LightGBM**
6. **CatBoost**

- **Accuracy** - Overall correctness of predictions.
- **Precision** - Correctly identified frauds out of all predicted frauds.
- **Recall** - Ability to detect all frauds.
- **F1-Score** - Harmonic mean of precision and recall.
- **ROC-AUC** - Model's ability to distinguish between fraud and non-fraud.

This heatmap visualizes the correlation between key features in the dataset.
<img src="/assets/images/Coorelation Heatmap for key features.png" alt="Correlation Heatmap for Key Features" width="500">


## Results

| **Model**                   | **Accuracy** | **Precision (0)** | **Precision (1)** | **Recall (0)** | **Recall (1)** | **F1-Score (0)** | **F1-Score (1)** | **ROC AUC** |
|------------------------------|--------------|--------------------|--------------------|----------------|----------------|-------------------|-------------------|-------------|
| **Random Forest**            | 0.8504       | 0.83              | 0.88              | 0.89          | 0.81          | 0.86             | 0.84             | 0.9241      |
| **Isolation Forest**         | 0.5327       | 0.52              | 0.86              | 0.99          | 0.06          | 0.68             | 0.11             | 0.5254      |
| **Hybrid (RF + IF)**         | 0.8410       | 0.82              | 0.87              | 0.88          | 0.80          | 0.85             | 0.83             | 0.8418      |
| **XGBoost**                  | 0.8809       | 0.87              | 0.89              | 0.90          | 0.86          | 0.88             | 0.88             | 0.9527      |
| **LightGBM**                 | 0.8723       | 0.86              | 0.89              | 0.90          | 0.85          | 0.88             | 0.87             | 0.9451      |
| **CatBoost**                 | 0.8630       | 0.84              | 0.88              | 0.89          | 0.83          | 0.87             | 0.86             | 0.9385      |
| **Hybrid (RF + XGB)**        | 0.8809       | 0.87              | 0.89              | 0.90          | 0.86          | 0.88             | 0.88             | 0.9525      |
| **XGBoost (Hyperparameter)** | 0.8858       | 0.87              | 0.90              | 0.91          | 0.86          | 0.89             | 0.88             | 0.9536      |
| **Random Forest (Hyperparameter)** | 0.8410 | 0.82              | 0.87              | 0.88          | 0.80          | 0.85             | 0.83             | 0.9233      |
| **Isolation Forest (Hyperparameter)** | 0.5376 | 0.52           | 0.86              | 0.99          | 0.06          | 0.68             | 0.11             | 0.5307      |

This bar chart compares the accuracy of different machine learning models.
<img src="/assets/images/Accuracy of Different Models.png" alt="Accuracy of Different Models" width="600">


## Conclusion

- **XGBoost** and **Hybrid Models** (RF + XGB) performed best with an accuracy of **88.58%** and ROC AUC of **95.36%**, as demonstrated by their **confusion matrices** and **ROC curves**.
- SHAP values revealed that **TransactionAmt** contributed **30%**, **card1** contributed **20%**, and **TransactionHour** contributed **15%** to the predictions. 

This SHAP summary plot highlights the most important features for the Random Forest model.
<img src="/assets/images/Shap summary plot for RF.png" alt="SHAP Summary Plot for RF" width="600">

This SHAP summary plot highlights the most important features for the XGBoost model.
<img src="/assets/images/Shap summary plot for XGB.png" alt="SHAP Summary Plot for XGB" width="600">

This SHAP dependence plot shows the interaction effect of `TransactionAmt` with the model's predictions.
<img src="/assets/images/Shap dependence plot.png" alt="SHAP Dependence Plot" width="500">


- Creating derived features like `TransactionHour` and `Amt_to_mean_ratio` significantly improved model predictions. Specifically, `TransactionHour` increased the Random Forest model's ROC AUC by **4.2%**, and `Amt_to_mean_ratio` contributed to a **6% improvement** in the F1-score across hybrid models.

## How to Run the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ShubhamAggarwl/finicial-fraud-detection
    cd fraud-detection
    ```

2. **Install Required Libraries**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost shap
    ```

3. **Run the Notebook**:
    Open and execute `finicial-fraud-detection.ipynb` to reproduce the results.

4. **Review Visualizations**:
    Examine confusion matrices, feature importance plots, and ROC curves to evaluate model performance.
