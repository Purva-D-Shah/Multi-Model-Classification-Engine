# Multi-Model Classification Engine: Telco Customer Churn

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy a machine learning classification engine to predict customer churn. By implementing and comparing multiple classification algorithms, this project aims to identify the most effective model for binary classification. The final phase involves deploying the models through an interactive web application using Streamlit, allowing users to evaluate performance metrics and upload test data for real-time predictions.

## b. Dataset Description
* **Dataset Name:** Telco Customer Churn
* **Source:** Kaggle
* **Problem Type:** Binary Classification
* **Instances:** 7,043 rows (Exceeds the 500 minimum requirement)
* **Features:** 21 columns (Exceeds the 12 minimum requirement)
* **Target Variable:** `Churn` (Yes/No)
* **Description:** This dataset contains customer-level information for a telecommunications company. It includes demographic data, account information (contract type, payment method, monthly charges), and the services each customer has signed up for. The goal is to predict which customers are likely to discontinue their service.

## c. Github Repository Link
https://github.com/Purva-D-Shah/Multi-Model-Classification-Engine

## d. Models used:

### Evaluation Metrics Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.787 | 0.832 | 0.621 | 0.516 | 0.564 | 0.428 |
| **Decision Tree** | 0.726 | 0.664 | 0.485 | 0.527 | 0.505 | 0.316 |
| **kNN** | 0.753 | 0.765 | 0.537 | 0.508 | 0.522 | 0.356 |
| **Naive Bayes** | 0.657 | 0.810 | 0.429 | 0.869 | 0.574 | 0.399 |
| **Random Forest (Ensemble)** | 0.785 | 0.815 | 0.625 | 0.473 | 0.539 | 0.408 |

### Model Performance Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall accuracy (0.787) and MCC (0.428), indicating it is the most balanced and reliable model for separating churners from non-churners in this dataset. |
| **Decision Tree** | Showed the lowest AUC (0.664) and MCC (0.316), suggesting it likely overfit the training data and struggles to generalize to the test set compared to the others. |
| **kNN** | Performed moderately well across all metrics but fell short of the predictive power seen in the linear and ensemble methods. |
| **Naive Bayes** | Yielded the highest recall (0.869) but the lowest precision and accuracy, meaning it correctly identifies most true churners but produces a high volume of false positives. |
| **Random Forest (Ensemble)** | Delivered very competitive accuracy and AUC, closely trailing Logistic Regression, though with a slightly lower recall score. |
| **Overall Winner for your dataset?** | **Logistic Regression**, due to its superior balance of Accuracy, AUC, and MCC, making it the most robust choice for this specific data. |
