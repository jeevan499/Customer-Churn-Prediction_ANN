import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
import tensorflow as tf


# Load the trained model and encoders
model = tf.keras.models.load_model('churn_model.h5')

# Load encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 

# Streamlit app
st.title("Customer Churn Prediction")

# User input
input_data = {
    'CreditScore': st.number_input("Credit Score", min_value=300, max_value=850, value=600),
    'Geography': st.selectbox("Geography", options=['France', 'Spain', 'Germany']),
    'Gender': st.selectbox("Gender", options=['Male', 'Female']),
    'Age': st.number_input("Age", min_value=18, max_value=100, value=30),
    'Tenure': st.number_input("Tenure (years)", min_value=0, max_value=10, value=3),
    'Balance': st.number_input("Balance", min_value=0.0, value=10000.0),
    'NumOfProducts': st.number_input("Number of Products", min_value=1, max_value=4, value=1),
    'HasCrCard': st.selectbox("Has Credit Card", options=[0, 1]),
    'IsActiveMember': st.selectbox("Is Active Member", options=[0, 1]),
    'EstimatedSalary': st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
}

# Prepare input data for prediction
input_df = pd.DataFrame([input_data])

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
# Label encode
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
# Scale the data
scaled_input = scaler.transform(input_df)   
# Predict the probability of churn
churn_probability = model.predict(scaled_input)[0][0]
st.write(f"Predicted Churn Probability: {churn_probability:.4f}")
if churn_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
