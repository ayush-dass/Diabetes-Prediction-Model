import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# Load the model and other objects using pickle
with open('diabetes_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

best_model = model_data['model']

confusion_matrix = model_data['confusion_matrix']

train_accuracy = model_data['train_accuracy']
test_accuracy = model_data['test_accuracy']

train_sizes = model_data['learning_curve']['train_sizes']
train_scores = model_data['learning_curve']['train_scores']
test_scores = model_data['learning_curve']['test_scores']

cv_scores = model_data['cross_validation']['cv_scores']
mean_cv_score = model_data['cross_validation']['mean_cv_score']

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Polynomial Features and Standardization
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    input_data_poly = poly.transform(input_data)
    input_data_scaled = scaler.transform(input_data_poly)
    prediction = best_model.predict(input_data_scaled)
    return prediction

# Main title
st.title('Diabetes Prediction App')

# Sidebar with input fields
st.sidebar.title('Input Features')
Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
SkinThickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
Insulin = st.sidebar.slider('Insulin', 0, 846, 30)
BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
Age = st.sidebar.slider('Age', 21, 81, 29)

# Button to predict
if st.sidebar.button('Predict'):
    prediction = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    if prediction[0] == 0:
        st.sidebar.success('No Diabetes')
    else:
        st.sidebar.error('Diabetes Detected')

# Model Explanation
st.write('### Model Explanation')
st.write("""This app uses an XGBoost classifier to predict whether a patient has diabetes or not. The model is trained on the Pima Indian Diabetes dataset, which contains various features like Glucose, Blood Pressure, BMI, etc. 
The dataset is preprocessed by handling missing values, outliers, and scaling. The model is trained using hyperparameters obtained from a random search and is evaluated using training and test accuracy, cross-validation score, confusion matrix, and learning curves.""")

# Display dataset
st.write('### Diabetes Dataset')
st.write(diabetes_data)

# Display model accuracy
st.write('### Model Evaluation')
st.success(f'Training Accuracy: {train_accuracy:.2f}')
st.success(f'Test Accuracy: {test_accuracy:.2f}')

# Display cross-validation score
st.write('### Cross-Validation Score and Mean')
st.success(f'Cross-Validation Scores: {cv_scores}')
st.success(f'Mean CV Accuracy: {mean_cv_score:.2f}')

# Display confusion matrix
st.write('### Confusion Matrix')
cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=['No Diabetes', 'Diabetes'])
fig, ax = plt.subplots(figsize=(6, 6))
cmd.plot(ax=ax)
st.pyplot(fig)

# Display learning curves
st.write('### Learning Curves')
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
plt.title('Learning Curves')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend()
st.pyplot(plt)

# Data visualization - Histogram
st.write('### Data visualization - Histogram')
selected_feature = st.selectbox('Select a feature to visualize:', diabetes_data.columns[:-1])
plt.figure(figsize=(8, 6))
sns.histplot(data=diabetes_data, x=selected_feature, hue='Outcome', kde=True, palette='Set1', alpha=0.7)
plt.title(f'Distribution of {selected_feature}')
plt.xlabel(selected_feature)
plt.ylabel('Count')
st.pyplot(plt)

# Data visualization - Scatter Plot
st.write('### Data visualization - Scatter Plot')
selected_features = st.multiselect('Select features:', diabetes_data.columns[:-1])

if len(selected_features) == 2:
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=diabetes_data, x=selected_features[0], y=selected_features[1], hue='Outcome', palette='Set1')
    plt.title(f'Scatter Plot: {selected_features[0]} vs {selected_features[1]}')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    st.pyplot(plt)

elif len(selected_features) > 2:
    st.error('You have selected more than two features! Consider removing some.')
else:
    st.warning('Select exactly two features to visualize.')

# Correlation Matrix
st.write('### Correlation Matrix')
plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
st.pyplot(plt)

