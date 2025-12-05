import flask
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load the pre-trained model and scaler --- 
# Ensure these files are in the same directory as your app.py
try:
    with open('diabetes_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure 'diabetes_prediction_model.pkl' and 'scaler.pkl' are in the same directory.")
    # Exit or handle the error appropriately if files are missing
    exit()

# --- Preprocessing components (must be consistent with training) ---
# These were calculated from the original training data
capping_bounds = {
    'bmi': {'lower': 14.705, 'upper': 38.505},
    'HbA1c_level': {'lower': 2.7, 'upper': 8.3},
    'blood_glucose_level': {'lower': 11.5, 'upper': 247.5}
}
gender_mapper = {'Male':0, 'Female':1, 'Other':2}
columns_to_standardize = ['age','bmi', 'HbA1c_level', 'blood_glucose_level']

# Feature names in the exact order the model expects
# This list should match `feature_names_for_prediction` from the notebook
feature_names_for_prediction = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'Encode_Gender', 'smoking_history_No Info', 'smoking_history_current',
    'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
    'smoking_history_not current'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert incoming JSON data to a pandas DataFrame
        # Ensure all expected fields are present and handle potential missing keys
        input_df = pd.DataFrame(json_, index=[0])

        # --- Apply the same preprocessing steps as in training ---

        # 1. Capping numerical features
        for col in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
            if col in input_df.columns:
                lb = capping_bounds[col]['lower']
                ub = capping_bounds[col]['upper']
                input_df[col] = np.where(input_df[col] < lb, lb, input_df[col])
                input_df[col] = np.where(input_df[col] > ub, ub, input_df[col])
            else:
                return jsonify({'error': f'Missing required input: {col}'}), 400

        # 2. Standard Scaling
        input_df[columns_to_standardize] = scaler.transform(input_df[columns_to_standardize])

        # 3. Encode gender
        if 'gender' in input_df.columns:
            input_df['Encode_Gender'] = input_df['gender'].map(gender_mapper).fillna(2) # Default to 'Other' if not found
            input_df.drop(columns=['gender'], inplace=True)
        else:
             return jsonify({'error': 'Missing required input: gender'}), 400

        # 4. One-hot encode smoking_history
        # Create columns for all smoking history categories, initialized to False
        for sh_col in [col for col in feature_names_for_prediction if 'smoking_history_' in col]:
            input_df[sh_col] = False
        
        if 'smoking_history' in input_df.columns:
            smoking_history_val = input_df['smoking_history'][0] # Get the single value
            if smoking_history_val == 'No Info':
                input_df['smoking_history_No Info'] = True
            else:
                col_name = f'smoking_history_{smoking_history_val}'
                if col_name in input_df.columns:
                    input_df[col_name] = True
                else:
                    # Handle unexpected smoking_history value
                    return jsonify({'error': f'Invalid smoking_history value: {smoking_history_val}'}), 400
            input_df.drop(columns=['smoking_history'], inplace=True)
        else:
            return jsonify({'error': 'Missing required input: smoking_history'}), 400
        
        # Ensure the final DataFrame has all features in the correct order
        # If any feature is missing from the input, it will be added with a default value (e.g., 0 for numerical, False for boolean)
        # This handles cases where some boolean smoking_history columns might not be explicitly created if smoking_history is not provided or is 'No Info'
        final_input = pd.DataFrame(columns=feature_names_for_prediction)
        final_input = pd.concat([final_input, input_df], ignore_index=True)
        final_input = final_input.fillna(False) # Fill missing boolean columns (smoking history) with False
        final_input = final_input.astype({col: bool for col in final_input.columns if 'smoking_history_' in col})
        
        # Convert to numpy array for prediction
        input_array = final_input[feature_names_for_prediction].values

        # Make prediction
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)

        # Return results
        result = {
            'prediction': int(prediction[0]),
            'probability_no_diabetes': float(prediction_proba[0][0]),
            'probability_diabetes': float(prediction_proba[0][1])
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development, use debug=True. For production, disable debug.
    # You might also want to specify host='0.0.0.0' to make it accessible externally
    app.run(debug=True, host='0.0.0.0', port=5000)
