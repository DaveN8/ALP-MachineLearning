from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_filename = 'model/random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Initialize scaler (you should fit this scaler on your training data)
scaler = RobustScaler()
# You may want to fit the scaler on your training data and save it as well

@app.route('/')
def home():
    return "Welcome to Water Potability Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        # Convert JSON data into DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Scale features (make sure to use the same scaler as used in training)
        input_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_scaled)
        
        # Return prediction result as JSON
        return jsonify({'Potability': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
