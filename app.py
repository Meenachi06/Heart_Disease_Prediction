import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

X, y = load_data()

# Train model (or load a pre-trained model)
@st.cache_resource
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("""
Enter the patient's information below to predict the likelihood of heart disease.
""")

# Input fields for all features
def user_input_features():
    age = st.number_input('Age', 20, 100, 50)
    sex = st.selectbox('Sex', [1, 0], format_func=lambda x: 'Male' if x==1 else 'Female')
    cp = st.selectbox('Chest Pain Type', [0,1,2,3], format_func=lambda x: ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'][x])
    trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)
    chol = st.number_input('Serum Cholestoral (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')
    restecg = st.selectbox('Resting ECG Results', [0,1,2])
    thalach = st.number_input('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')
    oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox('Slope of Peak Exercise ST', [0,1,2])
    ca = st.selectbox('Number of Major Vessels (0-3)', [0,1,2,3])
    thal = st.selectbox('Thalassemia', [1,2,3], format_func=lambda x: {1:'Normal',2:'Fixed Defect',3:'Reversible Defect'}[x])
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"High risk of heart disease! (Probability: {probability:.2%})")
    else:
        st.success(f"Low risk of heart disease. (Probability: {probability:.2%})")

st.markdown("---")
st.markdown("<small>Made with ❤️ using Streamlit</small>", unsafe_allow_html=True)
