# app.py
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Global variable for class names
class_names = None

# Model training function (runs once at startup)
def train_model():
    global class_names
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model to disk
    joblib.dump(model, 'iris_model.pkl')
    
    # Set class names globally
    class_names = iris.target_names

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request body
        data = request.get_json()
        
        # Check for required features
        features = ['x1', 'x2', 'x3', 'x4']
        for feature in features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Convert input data to the format expected by the model
        input_data = np.array([[
            float(data['x1']), 
            float(data['x2']), 
            float(data['x3']), 
            float(data['x4'])
        ]])
        
        # Load the model and make a prediction
        model = joblib.load('iris_model.pkl')
        prediction = model.predict(input_data)[0]
        
        # Return the result
        result = {
            'class_id': int(prediction),
            'class_name': class_names[prediction]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Train the model when the app starts
train_model()

# No app.run() here—Gunicorn will handle it