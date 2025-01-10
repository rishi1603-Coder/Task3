from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    # This will render the HTML form when you visit the root URL.
    return render_template('house.html')  # Ensure you have a house.html file in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract features from the request
        features = np.array(data['features']).reshape(1, -1)
        # Predict house price
        prediction = model.predict(features)
        # Return prediction as JSON
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        # Handle any errors during prediction
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
