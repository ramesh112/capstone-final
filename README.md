# ML Pipeline for Credit Card Fraud Detection

This project builds and evaluates a supervised **machine-learning** pipeline for credit card fraud detection on a large synthetic card-transaction dataset of 500,000 rows, with an empirical fraud rate of about 8.75%.

The focus is on modeling highly imbalanced, noisy transaction data and comparing linear and tree-based classifiers using metrics that emphasize fraud recall and ranking quality rather than raw accuracy.

---

## Project Overview

The notebook `capstone_final_banking_fraud_detection.ipynb` walks through a complete end‑to‑end workflow:

- Data loading and sanity checks  
- Exploratory data analysis for skewness, outliers, and class imbalance  
- Feature engineering (capping, log transforms, and interactions)  
- Baseline and advanced models (Logistic Regression, Decision Tree, Random Forest, ensembles)  
- Cross‑validation and hyperparameter tuning  
- Model evaluation with ROC AUC, precision/recall, confusion matrices, and per‑class metrics  

The goal is to detect a high share of fraudulent transactions while keeping false positives at a manageable level for operational use.

---

## Dataset

The dataset (e.g., `data/card_data.csv`) contains 500,000 card transactions with the following main columns:

- `distance_from_home` – Distance between the transaction location and the cardholder’s home.  
- `distance_from_last_transaction` – Distance from the location of the previous transaction.  
- `ratio_to_median_purchase_price` – Ratio of the current purchase amount to the cardholder’s median purchase price.  
- `repeat_retailer` – 1 if the merchant has been used before by this cardholder, 0 otherwise.  
- `used_chip` – 1 if the card’s chip was used, 0 otherwise.  
- `used_pin_number` – 1 if a PIN was used, 0 otherwise.  
- `online_order` – 1 for online card‑not‑present transactions, 0 otherwise.  
- `fraud` – Target label: 1 for fraudulent, 0 for legitimate.  

There are no missing values; all feature columns are numeric, with several continuous variables showing strong right skew and extreme outliers.

---

## Feature Engineering

To stabilize training and expose useful risk patterns, the notebook performs several feature‑engineering steps:

- **Outlier capping**  
  - Cap `distance_from_home`, `distance_from_last_transaction`, and `ratio_to_median_purchase_price` at their 99th percentile values to reduce the influence of extreme outliers.  

- **Log transforms**  
  - Create log‑scaled versions of key skewed features:  
    - `log_distance_from_home` = `log1p(distance_from_home)`  
    - `log_distance_from_last_transaction` = `log1p(distance_from_last_transaction)`  
    - `log_ratio_to_median_purchase_price` = `log1p(ratio_to_median_purchase_price)`  

- **Interaction features**  
  - `online_and_far` = `online_order * log_distance_from_home` to capture high risk when online transactions occur far from home.  
  - `high_ratio_online` = `online_order * log_ratio_to_median_purchase_price` to flag online transactions with unusually high amounts relative to the customer’s median spend.  

The final feature matrix `X` includes the original behavior features plus log‑transformed and interaction features, and `y` holds the binary target `fraud`.

---

## Modeling Approach

### Train/Test Split and Baseline

- The data is split into train and test sets with `train_test_split`, using:
  - `test_size=0.2` (20% hold‑out)  
  - `stratify=y` to preserve the fraud/non‑fraud ratio in both splits  
  - A fixed `random_state` for reproducibility.  

- A baseline `LogisticRegression` model is trained with:
  - `max_iter=1000` to ensure convergence  
  - `class_weight='balanced'` to directly handle class imbalance in the loss function.  

The baseline logistic regression achieves a ROC AUC of about 0.98 on the test set, with high recall on the fraud class when using the default 0.5 threshold.

### Multiple Models on Same Split

The notebook then fits several models on the same train/test split:

- Logistic Regression (baseline)  
- Decision Tree (`max_depth=5`, `class_weight='balanced'`)  
- Random Forest (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `class_weight='balanced'`, `n_jobs=-1`)  
- AdaBoost (`n_estimators=200`, `learning_rate=0.1`)  

For each model, it reports:

- ROC AUC  
- Confusion matrix (TN, FP, FN, TP)  
- Full classification report (precision, recall, F1, support by class)  

### Cross‑Validation and Hyperparameter Tuning

To robustly compare models and tune hyperparameters, the notebook uses 5‑fold cross‑validation (StratifiedKFold):

- **Models under CV**  
  - Logistic Regression (`class_weight='balanced'`)  
  - Random Forest (same hyperparameters as above)  

- **Metric**  
  - ROC AUC (`scoring='roc_auc'`) to emphasize ranking quality across thresholds.  

Cross‑validated results show:

- Logistic Regression: mean ROC AUC ~0.98 with very low variance across folds.  
- Random Forest: mean ROC AUC essentially 1.0, indicating near‑perfect separation on this dataset.  

### Grid Search

A narrower GridSearchCV is run to validate and refine hyperparameters:

- **Random Forest grid** (examples):  
  - `max_depth`: `[None, 10]`  
  - `max_features`: `['sqrt', 'log2']`  
  - `min_samples_leaf`: `[1, 3]`  

- **Logistic Regression grid**:  
  - `C`: `[0.1, 1.0, 10.0]`  
  - `penalty`: `['l2']` (with `solver='liblinear'`)  

Findings:

- Best Random Forest configuration uses `max_depth=None`, `max_features='sqrt'`, and `min_samples_leaf=3`, with CV ROC AUC still essentially 1.0.  
- Best Logistic Regression uses moderately regularized L2 (`C=0.1`), with AUC similar to the baseline and still below Random Forest.  

---

## Evaluation and Key Results

Key evaluation insights from the notebook:

- **Class imbalance**  
  - The fraud rate is ~8.75%, so naive accuracy is not meaningful; evaluation focuses on fraud recall, precision, ROC AUC, and precision‑recall behavior.  

- **Baseline Logistic Regression**  
  - ROC AUC ≈ 0.98 on the held‑out test set.  
  - High recall on fraud (≈0.95) at the default threshold, with moderate precision, reflecting many detected frauds but also non‑trivial false positives.  

- **Tree‑based models (Decision Tree, Random Forest)**  
  - Single Decision Tree with moderate depth already delivers very high ROC AUC (~1.0) and excellent fraud recall.  
  - Random Forest further improves robustness and maintains essentially perfect AUC under both single split and cross‑validation, confirming strong non‑linear signal in the engineered features.  

- **Feature importance**  
  - Capped/log versions of `distance_from_home`, `distance_from_last_transaction`, and `ratio_to_median_purchase_price`, along with `online_order` and `repeat_retailer`, emerge as the most influential predictors in the Random Forest.  

---

## How to Run

1. **Clone or copy the project**  
   P
