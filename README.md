# RABTECH Academy – Task 4: Model Comparison & Hyperparameter Tuning

## Project Overview

This project is part of the RABTECH Academy Artificial Intelligence & Machine Learning program.

The objective of Task 4 is to train multiple supervised classification models, optimize their hyperparameters using cross-validation, compare their performance, and select a champion model.

## Dataset

The project uses the Adult Income dataset.

The objective is to predict whether a person's income is:

- `0` = `<=50K`
- `1` = `>50K`

The dataset contains:

- 32,561 records
- 14 input features
- 1 target variable

The data contains both numerical and categorical features.

## Models

Four classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## Preprocessing

The preprocessing workflow uses Scikit-Learn pipelines.

### Numerical Features

- Missing values are handled using median imputation.
- Features are standardized using `StandardScaler`.

### Categorical Features

- Missing values are handled using most-frequent imputation.
- Categorical values are converted using `OneHotEncoder`.
- Unknown categories are handled using `handle_unknown="ignore"`.

The preprocessing is included inside each model pipeline to help prevent data leakage.

## Hyperparameter Optimization

`GridSearchCV` was used to optimize the models.

A **5-fold Stratified K-Fold cross-validation** strategy was used.

ROC-AUC was used as the scoring metric during hyperparameter optimization.

The tuned models were then evaluated on the held-out test set.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

ROC-AUC curves were also generated to visually compare the models.

## Champion Model

The champion model was selected automatically based on the highest ROC-AUC score on the evaluation results.

The selected model was saved as:

```text
champion_model.pkl
