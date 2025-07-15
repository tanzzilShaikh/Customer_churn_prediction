import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib


data = pd.read_csv(r"C:\Users\ASUS\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Cleaning
# 'Churn' column to binary (1 for 'Yes', 0 for 'No')
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

#replace empty strings with NaN and convert to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

# Convert categorical columns
data = pd.get_dummies(data, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                     'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Dropping unnecessary column
data = data.drop(['customerID'], axis=1)

# Splitting the data
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

# Random Forest Model with Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("\nRandom Forest Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.2f}")

# Save the Random Forest model
joblib.dump(rf_model, 'churn_model.pkl')

# Load and use the saved model
loaded_model = joblib.load('churn_model.pkl')
sample_data = X_test_scaled[0:1]
prediction = loaded_model.predict(sample_data)
print("\nSample Prediction:")
print("Churn" if prediction[0] == 1 else "No Churn")