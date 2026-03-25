🏠 House Price Prediction

The House Price Prediction project focuses on predicting housing prices using machine learning techniques.
By leveraging powerful Python libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, and Seaborn, this project builds a complete end-to-end ML pipeline, including data preprocessing, model selection, and hyperparameter tuning.

📌 Project Overview

The goal of this project is to develop a machine learning model that can accurately predict median house prices based on various features like location, population, income, and housing characteristics.

This project demonstrates a real-world ML workflow, including:

Data analysis
Feature engineering
Model comparison
Hyperparameter tuning
Final prediction system

Such solutions are widely used in real estate, finance, and investment decision-making.

🔑 Key Features
📊 Data Collection and Processing
Uses the California Housing dataset (housing.csv)
Features include:
Longitude & Latitude
Housing Median Age
Total Rooms & Bedrooms
Population & Households
Median Income
Ocean Proximity
Data is processed using Pandas and NumPy
📈 Data Visualization
Performed detailed Exploratory Data Analysis (EDA):
Histograms for feature distribution
Count plots for categorical variables
Correlation heatmap
Target variable distribution
Visualizations created using Matplotlib and Seaborn
🧹 Data Preprocessing
Handled missing values using SimpleImputer
Scaled numerical features using StandardScaler
Encoded categorical features using OneHotEncoder
Built a ColumnTransformer pipeline for structured preprocessing
🔀 Train-Test Split
Dataset split into:
Training set (80%)
Testing set (20%)
Ensures proper evaluation on unseen data
🤖 Model Building and Comparison

Multiple machine learning models were trained and compared:

Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
HistGradientBoosting Regressor

Used:

Pipeline for clean workflow
K-Fold Cross Validation (k=5) for reliable comparison
⚡ Hyperparameter Tuning
Applied GridSearchCV to optimize the best model
Tuned parameters like:
Learning rate
Max depth
Leaf nodes
Regularization
📏 Model Evaluation

Model performance evaluated using:

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

Also performed:

Residual analysis
Error distribution visualization
🔮 Prediction System

Built a custom prediction function that:

Takes user input features
Returns predicted house price

This makes the model reusable for real-world applications.
