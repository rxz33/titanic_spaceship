from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (adjust filename if needed)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return "Titanic Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features (ensure this matches your model)
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    return jsonify({
        'prediction': int(prediction[0])  # Assuming binary classification
    })

if __name__ == '__main__':
    app.run(debug=True)
