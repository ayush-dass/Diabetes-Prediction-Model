import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

# Display confusion matrix using Plotly
st.write('### Confusion Matrix')
z = confusion_matrix
x = ['No Diabetes', 'Diabetes']
y = ['No Diabetes', 'Diabetes']
z_text = [[str(y) for y in x] for x in z]

fig = go.Figure(data=go.Heatmap(
                   z=z,
                   x=x,
                   y=y,
                   hoverongaps=False,
                   colorscale='Blues',
                   text=z_text,
                   texttemplate="%{text}",
                   textfont={"size":15}))

fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='True Label')
st.plotly_chart(fig)

# Display learning curves using Plotly
st.write('### Learning Curves')
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1), mode='lines', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(test_scores, axis=1), mode='lines', name='Validation Accuracy'))
fig.update_layout(title='Learning Curves', xaxis_title='Training Examples', yaxis_title='Accuracy')
st.plotly_chart(fig)

# Data visualization - Histogram using Plotly
st.write('### Data visualization - Histogram')
selected_feature = st.selectbox('Select a feature to visualize:', diabetes_data.columns[:-1])
fig = px.histogram(diabetes_data, x=selected_feature, color='Outcome', barmode='overlay', nbins=50)
fig.update_layout(title=f'Distribution of {selected_feature}')
st.plotly_chart(fig)

# Data visualization - Scatter Plot using Plotly
st.write('### Data visualization - Scatter Plot')
selected_features = st.multiselect('Select features:', diabetes_data.columns[:-1], default=['Pregnancies', 'Glucose'])

if len(selected_features) == 2:
    # Scatter Plot
    fig = px.scatter(diabetes_data, x=selected_features[0], y=selected_features[1], color='Outcome', title=f'Scatter Plot: {selected_features[0]} vs {selected_features[1]}')
    st.plotly_chart(fig)
elif len(selected_features) > 2:
    st.error('You have selected more than two features! Consider removing some.')
else:
    st.warning('Select exactly two features to visualize.')

# Correlation Matrix using Plotly
st.write('### Correlation Matrix')
corr = diabetes_data.corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
fig.update_layout(title='Correlation Matrix')
st.plotly_chart(fig)
