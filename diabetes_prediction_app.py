import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Standardize the data for SVM and scale the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
y = diabetes_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    input_data_scaled = scaler.transform(input_data)
    prediction = svm_model.predict(input_data_scaled)
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
st.write("""
Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification tasks.
In this app, we have trained an SVM model with a linear kernel using the Pima Indians Diabetes dataset.
         
The model aims to predict the likelihood of an individual having diabetes based on various health features such as 
glucose level, BMI, age, etc.
The model learns patterns from the historical data to make predictions on new, unseen data.
""")

# Display dataset
st.write('### Diabetes Dataset')
st.write(diabetes_data)

# Display model accuracy
st.write('### Model Accuracy')
accuracy = accuracy_score(svm_model.predict(X_test), y_test)
st.success(f'Model Accuracy: {accuracy:.2f}')

# Data visualization - Histogram
st.write('### Data visualization - Histogram')
selected_feature = st.selectbox('Select a feature to visualize:', diabetes_data.columns[:-1])
plt.figure(figsize=(8, 6))
sns.histplot(data=diabetes_data, x=selected_feature, hue='Outcome', kde=True, palette='Set1', alpha=0.7)
plt.title(f'Distribution of {selected_feature}')
plt.xlabel(selected_feature)
plt.ylabel('Count')
st.pyplot(plt)

# Data visualization - Scatter Plot and Correlation Matrix
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

# Display feature importance
st.write('### Feature Importance')
coef_values = svm_model.coef_[0]
feature_importance = pd.DataFrame({'Feature': diabetes_data.columns[:-1], 'Coefficient': coef_values})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

st.write(feature_importance)

