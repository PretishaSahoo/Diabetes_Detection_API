from flask import Flask
from flask import request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), '../model/diabetes_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def welcome():
    return "Diabetes Detection API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                data['SkinThickness'], data['Insulin'], data['BMI'],
                data['DiabetesPedigreeFunction'], data['Age']]

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    result = {
        'prediction': int(prediction[0]), 
        'message': 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)