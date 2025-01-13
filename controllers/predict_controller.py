from flask import request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# Load the trained model and scaler
model_filename = 'model/best_model.pkl'
scaler_filename = 'model/robust_scaler.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

def home():
    return render_template('index.html')

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
        
        # Scale features using the loaded scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_scaled)
        
        # Determine result
        result = 'Potable' if prediction[0] == 1 else 'Not Potable'
        
        return render_template('result.html', result=result, input_data=data)
    
    except Exception as e:
        return render_template('index.html', error=str(e))
