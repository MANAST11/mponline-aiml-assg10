import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model at startup
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if not os.path.exists(model_path):
    print(f"Error: Model file 'model.pkl' not found at {model_path}.", file=sys.stderr)
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        model = None

# Feature list in the exact order required by the model
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def home():
    """Renders the interactive web application interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    REST API endpoint for heart disease prediction.
    Accepts patient details as JSON input and returns the prediction.
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded on the server. Please check server logs."
        }), 500

    # Ensure payload is JSON
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Invalid request content-type. Expected 'application/json'."
        }), 400

    data = request.get_json()

    # Extract features in the correct order
    features_list = []
    
    # Support direct key-value mapping (recommended)
    try:
        if 'features' in data and isinstance(data['features'], list):
            # Support list format: {"features": [52, 1, 0, ...]}
            if len(data['features']) != len(FEATURE_NAMES):
                return jsonify({
                    "status": "error",
                    "message": f"Expected {len(FEATURE_NAMES)} features, but got {len(data['features'])}."
                }), 400
            features_list = [float(x) for x in data['features']]
        else:
            # Support object format: {"age": 52, "sex": 1, ...}
            missing_features = [f for f in FEATURE_NAMES if f not in data]
            if missing_features:
                return jsonify({
                    "status": "error",
                    "message": f"Missing features in JSON body: {missing_features}"
                }), 400
            
            for feature in FEATURE_NAMES:
                features_list.append(float(data[feature]))
                
    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid feature value type. All features must be numeric. Details: {str(e)}"
        }), 400

    # Make prediction
    try:
        import pandas as pd
        features_df = pd.DataFrame([features_list], columns=FEATURE_NAMES)
        prediction_val = int(model.predict(features_df)[0])
        
        # Get probability if available
        probability = None
        if hasattr(model, "predict_proba"):
            prob_scores = model.predict_proba(features_df)[0]
            probability = float(prob_scores[prediction_val])

        # Map binary outcome to user-facing strings
        prediction_str = "Heart Disease Detected" if prediction_val == 1 else "No Heart Disease Detected"

        response = {
            "prediction": prediction_str,
            "status": "success"
        }
        
        if probability is not None:
            response["probability"] = round(probability, 4)
            response["risk_score_percent"] = round((prob_scores[1] * 100), 2) if prediction_val == 1 else round((1 - prob_scores[0]) * 100, 2)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error running prediction model: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
