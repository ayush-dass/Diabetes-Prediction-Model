import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Handle Missing Values
missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='mean')
diabetes_data[missing_values] = imputer.fit_transform(diabetes_data[missing_values])

# Outlier Detection and Treatment
def handle_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data.loc[data[feature] < lower_bound, feature] = lower_bound
    data.loc[data[feature] > upper_bound, feature] = upper_bound

# Handle outliers for 'Insulin' and 'BMI'
handle_outliers(diabetes_data, 'Insulin')
handle_outliers(diabetes_data, 'BMI')

# Separate features and target variable
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=2)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)

# Define XGBoost classifier with hyperparameters from random search
best_xgb_model = xgb.XGBClassifier(colsample_bytree=0.8795639148943363,
                                   learning_rate=0.025048646490632924,
                                   max_depth=7,
                                   reg_alpha=5,
                                   reg_lambda=6,
                                   subsample=0.7123807356822325,
                                   random_state=2,
                                   n_estimators=100)
                             
best_xgb_model.fit(X_train, y_train)

# Evaluate the best model from random search
train_accuracy = best_xgb_model.score(X_train, y_train)
test_accuracy = best_xgb_model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Predictions on the test set
y_pred = best_xgb_model.predict(X_test)

# Probabilities for positive class (class 1) from the model
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc}")

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Cross-validation score
cv_scores = cross_val_score(best_xgb_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
mean_cv_accuracy = np.mean(cv_scores)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {mean_cv_accuracy}")

# Calculate learning curve
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    best_xgb_model, X_resampled, y_resampled, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), return_times=True)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Save the model and other data to a pickle file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_xgb_model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'roc_auc_score': roc_auc,
        'confusion_matrix': conf_matrix,
        'learning_curve': {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        },
        'cross_validation': {
            'cv_scores': cv_scores,
            'mean_cv_score': mean_cv_accuracy
        }
    }, f)

print("Trained Model data saved to diabetes_model.pkl")
