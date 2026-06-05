# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. This project predicts whether a customer is likely to leave the company based on customer demographics, service usage, billing information, and contract details.

The objective is to help businesses identify at-risk customers and take proactive retention measures.

---

## Business Problem

Customer acquisition is often more expensive than customer retention. By identifying customers who are likely to churn, companies can:

* Improve customer retention
* Reduce revenue loss
* Optimize marketing efforts
* Increase customer lifetime value

---

## Dataset Information

Dataset: Telco Customer Churn Dataset

Features include:

* Customer demographics
* Contract type
* Internet services
* Payment methods
* Monthly charges
* Total charges
* Tenure

Target Variable:

* Churn (Yes / No)

---

## Project Workflow

1. Data Cleaning
2. Missing Value Treatment
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Feature Engineering
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Hyperparameter Tuning
10. Feature Importance Analysis
11. Model Deployment using Streamlit

---

## Machine Learning Models Evaluated

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

---

## Final Model Performance

Best Model: Tuned Random Forest Classifier

Metrics:

* Accuracy: 79.10%
* Precision: 64.18%
* Recall: 48.40%
* F1 Score: 55.18%
* ROC-AUC: 83.04%

---

## Important Features

Top factors influencing customer churn:

* Total Charges
* Monthly Charges
* Tenure
* Internet Service
* Payment Method
* Online Security
* Contract Type

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Joblib
* Jupyter Notebook

---

## Project Files

* customer_churn_prediction.ipynb
* churn.py
* customer_churn_model.pkl
* feature_columns.pkl
* requirements.txt

---

## Streamlit Deployment

The trained machine learning model was deployed using Streamlit to provide real-time churn predictions through an interactive web application.

---

## Author

Shiva K

Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning
