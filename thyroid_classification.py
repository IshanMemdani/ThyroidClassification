import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('thyroid_data.csv')

# Check the first few rows to ensure the data is loaded correctly
print(data.head())
print(data.columns)

# Select the features (age, gender, TSH, T3, etc.)
X = data[['age', 'gender', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid', 
          'sick', 'pregnant', 'thyroid_surgery', 'i131_treatment', 'query_hypothyroid', 
          'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
          'TSH', 'T3', 'TT4', 'T4U', 'FTI']]

# Handle categorical columns such as 'gender', 'on_thyroxine', etc. using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Target column (class1: the target labels for thyroid classification)
y = data['class1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# --- Support Vector Machine Model ---
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Feature selection using Recursive Feature Elimination (RFE)
rf_selector = RFE(rf_model, n_features_to_select=10)  # Select top 10 features
rf_selector = rf_selector.fit(X_train, y_train)

# Transform the features based on RFE selection
X_train_selected = rf_selector.transform(X_train)
X_test_selected = rf_selector.transform(X_test)

# Retrain the models with the selected features
rf_model.fit(X_train_selected, y_train)
svm_model.fit(X_train_selected, y_train)

# --- Hybrid Model Prediction ---
rf_pred = rf_model.predict(X_test_selected)
svm_pred = svm_model.predict(X_test_selected)

# Combine the predictions from both models (for hybrid approach)
# Simple majority voting or other ensemble techniques can be applied
hybrid_pred = np.round((rf_pred + svm_pred) / 2).astype(int)

# Evaluate the models
print("Random Forest Model Evaluation:")
print(classification_report(y_test, rf_pred))

print("SVM Model Evaluation:")
print(classification_report(y_test, svm_pred))

print("Hybrid Model Evaluation (Random Forest + SVM):")
print(classification_report(y_test, hybrid_pred))

# Optionally, save the models
import pickle
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save the scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)