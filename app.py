import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the model, scaler, and feature selector
with open('hybrid_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
    scaler = pickle.load(f)
    selector = pickle.load(f)

# Define a function to encode categorical variables
def encode_data(data):
    # Define the categorical columns to encode (based on the columns in the dataset)
    categorical_columns = [
        'gender', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid', 
        'sick', 'pregnant', 'thyroid_surgery', 'i131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych'
    ]
    
    # Create a DataFrame to hold the input data
    input_data = pd.DataFrame([data])
    
    # Encode the categorical variables (using pandas get_dummies for one-hot encoding)
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns)
    
    # Ensure the column order matches the training data
    # Here, we assume you have the same columns used during training
    missing_cols = set(scaler.feature_names_in_) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0

    # Reorder the columns to match the trained model
    input_data_encoded = input_data_encoded[scaler.feature_names_in_]
    
    # Return the encoded data
    return input_data_encoded

@app.route('/')
def home():
    return "Thyroid Disorder Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Preprocess the data
        input_data = encode_data(data)
        
        # Preprocess the input data (scaling and feature selection)
        input_data_scaled = scaler.transform(input_data)
        input_data_selected = selector.transform(input_data_scaled)
        
        # Predict using the loaded SVM model
        prediction = svm_model.predict(input_data_selected)
        
        # Convert prediction to a standard Python data type (e.g., str)
        result = str(prediction[0])

        # Return the prediction as a JSON response
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
