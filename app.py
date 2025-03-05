# app.py
import joblib
import numpy as np
import logging
import traceback
import json
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Configure logging to focus on errors
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - ERROR: %(message)s'
)

# Global variable for class names
class_names = None

# Model training function (runs once at startup)
def train_model():
    global class_names
    
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model to the models directory
    model_path = '/app/models/iris_model.pkl'
    try:
        joblib.dump(model, model_path)
        app.logger.error(f"Model successfully saved to {model_path}")
    except Exception as e:
        app.logger.error(f"Error saving model: {str(e)}")
        app.logger.error(traceback.format_exc())
        # Fallback to current directory if /app/models is not accessible
        joblib.dump(model, 'iris_model.pkl')
    
    # Store class names for later use
    class_names = iris.target_names

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request body
        data = request.get_json()
        
        # Log request data in case of errors
        app.logger.error(f"Request data: {json.dumps(data, default=str)}")
        
        # Check for required features
        features = ['x1', 'x2', 'x3', 'x4']
        for feature in features:
            if feature not in data:
                error_msg = f'Missing feature: {feature}'
                app.logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
        
        # Convert input data to the format expected by the model
        try:
            input_data = np.array([[
                float(data['x1']), 
                float(data['x2']), 
                float(data['x3']), 
                float(data['x4'])
            ]])
        except ValueError as ve:
            error_msg = f'Error converting input data: {str(ve)}'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Load the model and make a prediction
        try:
            # Try to load from the models directory first
            try:
                model = joblib.load('/app/models/iris_model.pkl')
            except:
                # Fallback to current directory
                model = joblib.load('iris_model.pkl')
                
            prediction = model.predict(input_data)[0]
        except Exception as model_error:
            error_msg = f'Model prediction failed: {str(model_error)}'
            app.logger.error(error_msg)
            app.logger.error(traceback.format_exc())  # Log the full stack trace
            return jsonify({'error': error_msg}), 500
        
        # Return the result
        result = {
            'class_id': int(prediction),
            'class_name': class_names[prediction]
        }
        
        return jsonify(result)
    
    except Exception as e:
        # Log the full error details including stack trace
        app.logger.error(f'Unexpected error in predict endpoint: {str(e)}')
        app.logger.error(traceback.format_exc())
        
        # Check for common issues
        if request.data == b'':
            return jsonify({'error': 'Empty request body'}), 400
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON format'}), 400
            
        # Return a generic error message to the client
        return jsonify({'error': str(e)}), 500

# Train the model when the app starts
train_model()

# No app.run() here—Gunicorn will handle it