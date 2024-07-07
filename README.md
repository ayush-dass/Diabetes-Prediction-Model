# Diabetes Prediction App

This is a Streamlit web application for predicting diabetes risk based on health features using an XGBoost model.

## About

This project is a web application built using Streamlit, a Python library for creating interactive web apps. It leverages an XGBoost model trained on the Pima Indians Diabetes dataset to predict the likelihood of an individual having diabetes based on various health features such as glucose level, BMI, age, etc.

## Demo

A live demo of the app can be found [here](https://ayush-dass-diabetes-prediction-m-diabetes-prediction-app-jdnxzb.streamlit.app/)

## Features

- Allows users to input health features like pregnancies, glucose level, blood pressure, etc.
- Predicts diabetes risk using an XGBoost model.
- Displays model accuracy, cross-validation scores, and learning curves.
- Provides data visualization with histograms, scatter plots, and correlation matrices.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/diabetes-prediction-app.git
```

2. Navigate to the project directory:

```bash
cd diabetes-prediction-app
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run diabetes_prediction_app.py
```

2. Open your web browser and go to `http://localhost:8501` to access the app.

3. Use the sidebar to input health features and click the "Predict" button to see the diabetes prediction.

## Model Training

The model was trained using the following steps:

1. Load the dataset and handle missing values using `SimpleImputer`.
2. Handle outliers for specific features.
3. Separate features and target variable.
4. Apply polynomial features and standardize the data.
5. Balance the dataset using SMOTE.
6. Split the data into training and testing sets.
7. Define and train an XGBoost classifier with hyperparameters obtained from a random search.
8. Evaluate the model using training accuracy, test accuracy, ROC-AUC score, classification report, cross-validation score, learning curves, and confusion matrix.

## Files

- `diabetes_prediction_app.py`: Main file to run the Streamlit app.
- `diabetes_model.pkl`: Pickle file containing the trained model and other data.
- `diabetes.csv`: Dataset used for training the model.
- `requirements.txt`: List of dependencies required to run the app.

## Requirements

- Python 3.7+
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- imbalanced-learn
- xgboost
- pickle