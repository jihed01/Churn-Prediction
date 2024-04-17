# Churn Prediction Project

## Overview
This project focuses on predicting customer churn based on various features described in the datasets `train.csv` and `test.csv`. The goal is to build a predictive model that forecasts the likelihood of a customer discontinuing their subscription.

## Table of Contents
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction and Submission](#prediction-and-submission)
- [Tools and Libraries](#tools-and-libraries)

## Data Description
The datasets include the following features:

| Column Name                 | Description                                                |
|-----------------------------|------------------------------------------------------------|
| AccountAge                  | The age of the user's account in months.                   |
| MonthlyCharges              | The amount charged to the user on a monthly basis.         |
| TotalCharges                | The total charges incurred by the user over the account's lifetime. |
| SubscriptionType            | The type of subscription chosen by the user (Basic, Standard, Premium). |
| PaymentMethod               | The method of payment used by the user.                    |
| PaperlessBilling            | Indicates whether the user has opted for paperless billing (Yes or No). |
| ContentType                 | The type of content preferred by the user (Movies, TV Shows, or Both). |
| MultiDeviceAccess           | Indicates whether the user has access to the service on multiple devices (Yes or No). |
| DeviceRegistered            | The type of device registered by the user (TV, Mobile, Tablet, or Computer). |
| ViewingHoursPerWeek         | The number of hours the user spends watching content per week. |
| ContentDownloadsPerMonth    | The number of content downloads by the user per month.     |
| GenrePreference             | The preferred genre of content chosen by the user.         |
| UserRating                  | The user's rating for the service on a scale of 1 to 5.    |
| SupportTicketsPerMonth      | The number of support tickets raised by the user per month.|
| Gender                      | The gender of the user (Male or Female).                   |
| WatchlistSize               | The number of items in the user's watchlist.               |
| ParentalControl             | Indicates whether parental control is enabled for the user (Yes or No). |
| SubtitlesEnabled            | Indicates whether subtitles are enabled for the user (Yes or No). |
| CustomerID                  | A unique identifier for each customer.                     |
| Churn                       | The target variable indicating whether a user has churned or not (1 for churned, 0 for not churned). |

## Exploratory Data Analysis (EDA)
We performed an initial analysis to understand the distributions of various features and their relationship with the target variable, Churn. Insights from EDA are used to guide feature selection and engineering.

### Visualizations
Visual distributions of features such as `AccountAge`, `MonthlyCharges`, and others help us understand the customer base and identify patterns that might indicate churn likelihood.

## Preprocessing
### Feature Engineering
- **Log Transformation:** Applied to `TotalCharges` to normalize its distribution.
- **Scaling:** StandardScaler is used to scale numerical features to ensure equal weighting in the model.

### Handling Categorical Data
- **One-Hot Encoding:** Applied to convert categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.

### Handling Imbalanced Data
- **SMOTE Technique:** Used to oversample the minority class in the dataset to balance the distribution of the target class.

## Model Training
### Model Selection
- **Logistic Regression:** Started with a simple model to establish a baseline.

### Hyperparameter Tuning
- **GridSearchCV:** Used to find the optimal parameters for the Logistic Regression model.

## Model Evaluation
- **ROC-AUC Score:** Used to evaluate model performance due to the class imbalance in the target variable.

## Prediction and Submission
Prepared `prediction_df` with `CustomerID` and `predicted_probability` reflecting the likelihood of churn. Ensured the dataframe had exactly 104,480 rows and 2 columns as required for submission.

## Tools and Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
