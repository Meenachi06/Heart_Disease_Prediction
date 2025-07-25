import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def add_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        }
        .block-container {
            padding: 2rem;
        }
        div[data-testid="stForm"] {
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            border-radius: 10px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border-radius: 0.5rem;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        .high-risk {
            background-color: #fee2e2;
            border: 2px solid #ef4444;
            color: #dc2626;
        }
        .low-risk {
            background-color: #dcfce7;
            border: 2px solid #22c55e;
            color: #16a34a;
        }
        </style>
    """, unsafe_allow_html=True)

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

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Apply custom styling
add_custom_style()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #1e40af; margin-bottom: 1rem;'>
        ❤️ Heart Disease Prediction
    </h1>
    <p style='text-align: center; font-size: 1.2em; color: #475569; margin-bottom: 2rem;'>
        Enter the patient's information below to predict the likelihood of heart disease.
    </p>
""", unsafe_allow_html=True)

# Create form for input
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
          
    with col1:
        st.markdown("##### Patient Demographics")
        age = st.number_input('Age', 20, 100, 50)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    
    with col2:
        st.markdown("##### Vital Measurements")
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
        thalach = st.number_input('Maximum Heart Rate', 60, 220, 150)
    
    with col3:
        st.markdown("##### Test Results")
        restecg = st.selectbox('Resting ECG Results', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        oldpeak = st.number_input('ST Depression', 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox('ST/Heart Rate Slope', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.number_input('Number of Major Vessels (0-3)', 0, 3, 0)
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Convert categorical inputs to numerical
    sex_dict = {'Male': 1, 'Female': 0}
    cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_dict = {'Yes': 1, 'No': 0}
    restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_dict = {'Yes': 1, 'No': 0}
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    
    # Create feature dictionary
    data = {
        'age': age,
        'sex': sex_dict[sex],
        'cp': cp_dict[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_dict[fbs],
        'restecg': restecg_dict[restecg],
        'thalach': thalach,
        'exang': exang_dict[exang],
        'oldpeak': oldpeak,
        'slope': slope_dict[slope],
        'ca': ca,
        'thal': thal_dict[thal]
    }
    
    # Make prediction
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Display result
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-box high-risk">
                <h2 style='font-size: 1.5em; margin-bottom: 0.5rem;'>⚠️ High Risk of Heart Disease</h2>
                <p style='font-size: 1.2em; margin-bottom: 0.5rem;'>Probability: {probability:.1%}</p>
                <p style='font-size: 0.9em; color: #991b1b;'>
                    Please consult with a healthcare professional for a thorough evaluation.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box low-risk">
                <h2 style='font-size: 1.5em; margin-bottom: 0.5rem;'>✅ Low Risk of Heart Disease</h2>
                <p style='font-size: 1.2em; margin-bottom: 0.5rem;'>Probability: {probability:.1%}</p>
                <p style='font-size: 0.9em; color: #166534;'>
                    Continue maintaining a healthy lifestyle!
                </p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #64748b; font-size: 0.9em;'>
            Made with ❤️ using Streamlit | For educational purposes only. Not for medical diagnosis.
        </p>
    </div>
""", unsafe_allow_html=True)

