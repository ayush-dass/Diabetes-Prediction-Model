# Diabetes Prediction App

This is a Streamlit web application for predicting diabetes risk based on health features using a Support Vector Machine (SVM) model.

## About

This project is a web application built using Streamlit, a Python library for creating interactive web apps. It leverages an SVM model trained on the Pima Indians Diabetes dataset to predict the likelihood of an individual having diabetes based on various health features such as glucose level, BMI, age, etc.

## Demo

A live demo of the app can be found [here](#)

## Features

- Allows users to input health features like pregnancies, glucose level, blood pressure, etc.
- Predicts diabetes risk using an SVM model.
- Displays model accuracy and feature importance.
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
