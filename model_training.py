import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = pd.read_csv('diabetes_data.csv')
print(f"Dataset shape: {data.shape}")
print(data.head())
print(data.isna().sum())
print(data.dtypes)

# Check proportion of target feature
print(data['diabetes'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# Select continuous columns
continuous_cols = ['age','bmi', 'HbA1c_level', 'blood_glucose_level']

# Create box plots for continuous columns
plt.figure(figsize=(20, 10))
for i, col in enumerate(continuous_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=data[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

data = cap_outliers_iqr(data, 'bmi')
data = cap_outliers_iqr(data, 'HbA1c_level')
data = cap_outliers_iqr(data, 'blood_glucose_level')

# Re-generate box plots to visualize the effect of capping
plt.figure(figsize=(15, 5))
for i, col in enumerate(['bmi', 'HbA1c_level', 'blood_glucose_level']):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=data[col])
    plt.title(f'Box Plot of {col} (After Capping)')
plt.tight_layout()
plt.show()

print(data['blood_glucose_level'].skew())
print('SKEWED VALUE IS IN BETWEEN -0.5 to +0.5 -> SO FAIRLY SYMMETRICAL')
print(data['gender'].unique())

# Encode datatype
gender_mapper = {'Male':0, 'Female':1, 'Other':2}
data['Encode_Gender'] = data['gender'].replace(gender_mapper)
data['Encode_Gender'] = pd.to_numeric(data['Encode_Gender'])

data['smoking_history'].value_counts()
data = pd.get_dummies(data, columns=['smoking_history'])
print(data.head())

columns_to_drop = [
    'gender'
]
data.drop(columns=columns_to_drop, inplace=True)

correation_with_diabetes = data.corr()['diabetes'].sort_values(ascending=False)
print(correation_with_diabetes)


# Initialize StandardScaler
scaler = StandardScaler()

# Columns to standardize
columns_to_standardize = ['age','bmi', 'HbA1c_level', 'blood_glucose_level']

# Apply StandardScaler to the selected columns
data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])

# Display the head of the DataFrame to show the standardized columns
print(data.head())

# Split DataFrame into X (train features) and y (predict features)
X = data.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13]].values
y = data['diabetes'].values

# Create train and test set
#Splitting the data into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20, random_state=42)

print('X TRAIN DATA ', X_train.shape)
print('Y TRAIN DATA ', y_train.shape)
print('X TEST DATA ', X_test.shape)
print('Y TEST DATA ', y_test.shape)

balancer = SMOTE(random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} with SMOTE...")

    pipeline = Pipeline(steps=[
        ('balancer', SMOTE(random_state=42)),  
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

    print(f"{name} trained successfully.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

print("\n--- Model Performance Evaluation ---")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

y_pred_test = trained_models['Gradient Boosting'].predict(X_test)
print(y_pred_test)


# Find the best performance model
best_accuracy = -1
best_model_name = None

for name, metrics in results.items():
    if metrics['Accuracy'] > best_accuracy:
        best_accuracy = metrics['Accuracy']
        best_model_name = name

print(f"The best performing model is '{best_model_name}' with an accuracy of {best_accuracy:.4f}")

# pickle file
import pickle
model_to_save = trained_models['Gradient Boosting']

# Define the filename for the pickle file
filename = 'diabetes_prediction_model.pkl'

# Save the model to a pickle file
with open(filename, 'wb') as file:
    pickle.dump(model_to_save, file)

print(f"Model successfully saved to {filename}")

# Save the scaler object
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

print(f"Scaler successfully saved to {scaler_filename}")

