from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_filename = 'model/best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the scaler (make sure to fit it on your training data)
scaler_filename = 'model/robust_scaler.pkl'
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            "ph": float(request.form['ph']),
            "Hardness": float(request.form['Hardness']),
            "Solids": float(request.form['Solids']),
            "Chloramines": float(request.form['Chloramines']),
            "Sulfate": float(request.form['Sulfate']),
            "Conductivity": float(request.form['Conductivity']),
            "Organic_carbon": float(request.form['Organic_carbon']),
            "Trihalomethanes": float(request.form['Trihalomethanes']),
            "Turbidity": float(request.form['Turbidity'])
        }
        
        # Convert to DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_scaled)
        
        # Return prediction result to the template
        result = 'Potable' if prediction[0] == 1 else 'Not Potable'
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
