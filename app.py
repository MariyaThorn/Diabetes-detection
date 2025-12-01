from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Load the trained model ---
try:
    with open('best_diabetes_model.pkl', 'rb') as file:
        best_model = pickle.load(file)
    print("Model 'best_diabetes_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("Error: 'best_diabetes_model.pkl' not found. Ensure the model is saved in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Re-create preprocessing components from original data ---
# This ensures preprocessing consistency with the training phase.
# We need to reload the original data to calculate capping bounds and fit scaler
# exactly as it was done during model training.

try:
    original_data = pd.read_csv("/content/drive/MyDrive/Aupp/Fall 2025/Cloud ML and Data Engineering/ML Model Deployment on EC2 and Access it/diabetes_data.csv")
except FileNotFoundError:
    print("Error: diabetes_data.csv not found. Please ensure the path is correct or the file is mounted.")
    exit()

# 1. Capping bounds calculation (from original data before any capping applied)
capping_bounds = {}
for col in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
    Q1 = original_data[col].quantile(0.25)
    Q3 = original_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    capping_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

# 2. Encode Gender and One-hot encode smoking_history for scaler fitting
gender_mapper = {'Male':0, 'Female':1, 'Other':2}
# Set the pandas option to opt-in to the future behavior for replace to avoid FutureWarning
pd.set_option('future.no_silent_downcasting', True)
original_data['Encode_Gender'] = original_data['gender'].replace(gender_mapper)
original_data['Encode_Gender'] = pd.to_numeric(original_data['Encode_Gender'])
original_data = pd.get_dummies(original_data, columns=['smoking_history'], dtype=bool)
original_data.drop(columns=['gender'], inplace=True)

# Apply capping to original_data (this is what happened during training prep)
for col in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
    lb = capping_bounds[col]['lower']
    ub = capping_bounds[col]['upper']
    original_data[col] = np.where(original_data[col] < lb, lb, original_data[col])
    original_data[col] = np.where(original_data[col] > ub, ub, original_data[col])

# 3. Fit the scaler on the preprocessed original_data
columns_to_standardize = ['age','bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
scaler.fit(original_data[columns_to_standardize])

# --- Define Prediction Function (similar to the one in the notebook but self-contained) ---
def predict_diabetes_api(age, gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history):
    user_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    for col in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
        lb = capping_bounds[col]['lower']
        ub = capping_bounds[col]['upper']
        user_data[col] = np.where(user_data[col] < lb, lb, user_data[col])
        user_data[col] = np.where(user_data[col] > ub, ub, user_data[col])

    user_data[columns_to_standardize] = scaler.transform(user_data[columns_to_standardize])

    encoded_gender = gender_mapper.get(gender, 2)

    input_smoking_features = {
        'smoking_history_current': False,
        'smoking_history_ever': False,
        'smoking_history_former': False,
        'smoking_history_never': False,
        'smoking_history_not current': False
    }
    if smoking_history != 'No Info':
        col_name = f'smoking_history_{smoking_history}'
        if col_name in input_smoking_features:
            input_smoking_features[col_name] = True

    input_df_for_pred = pd.DataFrame({
        'age': user_data['age'][0],
        'hypertension': user_data['hypertension'][0],
        'heart_disease': user_data['heart_disease'][0],
        'bmi': user_data['bmi'][0],
        'HbA1c_level': user_data['HbA1c_level'][0],
        'blood_glucose_level': user_data['blood_glucose_level'][0],
        'Encode_Gender': encoded_gender
    }, index=[0])

    input_df_for_pred = pd.concat([
        input_df_for_pred,
        pd.DataFrame([input_smoking_features])
    ], axis=1)

    final_features_order = [
        'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'Encode_Gender',
        'smoking_history_current', 'smoking_history_ever', 'smoking_history_former',
        'smoking_history_never', 'smoking_history_not current'
    ]
    input_array = input_df_for_pred[final_features_order].values

    prediction = best_model.predict(input_array)
    prediction_proba = best_model.predict_proba(input_array)

    return prediction[0], prediction_proba[0]

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features from the request data
        age = data['age']
        gender = data['gender'] # 'Male', 'Female', 'Other'
        hypertension = data['hypertension']
        heart_disease = data['heart_disease']
        bmi = data['bmi']
        hba1c_level = data['HbA1c_level']
        blood_glucose_level = data['blood_glucose_level']
        smoking_history = data['smoking_history'] # 'No Info', 'never', 'former', 'current', 'not current', 'ever'

        # Get prediction and probabilities
        predicted_class, probabilities = predict_diabetes_api(
            age, gender, hypertension, heart_disease, bmi,
            hba1c_level, blood_glucose_level, smoking_history
        )

        result = {
            'prediction': int(predicted_class),
            'probability_no_diabetes': float(probabilities[0]),
            'probability_diabetes': float(probabilities[1]),
            'risk_level': 'HIGH RISK OF DIABETES' if predicted_class == 1 else 'LOW RISK OF DIABETES'
        }

        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Missing data field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


