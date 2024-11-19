import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and label encoders
model = pickle.load(open('model.pkl', 'rb'))
lb1 = pickle.load(open('lb1.pkl', 'rb'))
lb2 = pickle.load(open('lb2.pkl', 'rb'))
lb3 = pickle.load(open('lb3.pkl', 'rb'))
lb4 = pickle.load(open('lb4.pkl', 'rb'))
lb5 = pickle.load(open('lb5.pkl', 'rb'))
lb6 = pickle.load(open('lb6.pkl', 'rb'))

# Streamlit setup
st.set_page_config(page_title="Student Grade Prediction", layout="centered")

# Add background animation of kids studying (CSS trick)
st.markdown("""
    <style>
    body {
        background: url('https://media.giphy.com/media/xT9IgM2O6yGT1TG4tq/giphy.gif') no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Student Grade Prediction")
st.write("""
    Welcome to the Student Grade Prediction app! 🎓
    Please input your details below, and the app will predict your final grade based on the provided information.
    """)

# Create a form to capture user input
with st.form(key='input_form'):
    school = st.selectbox("School", options=["GP", "MS"], index=0)
    reason = st.selectbox("Reason for choosing the school", options=["course", "home", "other", "reputation"])
    schoolsup = st.radio("School Support", options=["no", "yes"], index=0)
    famsup = st.radio("Family Support", options=["no", "yes"], index=0)
    paid = st.radio("Paid Extra Classes", options=["no", "yes"], index=0)
    internet = st.radio("Access to Internet", options=["no", "yes"], index=0)
    age = st.number_input("Age", min_value=15, max_value=22, value=18)
    Medu = st.slider("Mother's Education", min_value=0, max_value=4, value=2)
    Fedu = st.slider("Father's Education", min_value=0, max_value=4, value=2)
    studytime = st.slider("Study Time", min_value=1, max_value=4, value=2)
    failures = st.slider("Number of Past Failures", min_value=0, max_value=4, value=1)
    famrel = st.slider("Family Relationships", min_value=1, max_value=5, value=4)
    freetime = st.slider("Free Time After School", min_value=1, max_value=5, value=3)
    goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=3)
    health = st.slider("Health Status", min_value=1, max_value=5, value=4)
    absences = st.number_input("Number of Absences", min_value=0, value=0)
    G1 = st.number_input("Grade for Period 1", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Grade for Period 2", min_value=0, max_value=20, value=12)

    submit_button = st.form_submit_button("Predict Grade")

# When form is submitted, process the inputs and make a prediction
if submit_button:
    # Prepare user input data
    user_data = pd.DataFrame({
        'school': [0 if school == "GP" else 1],
        'reason': [["course", "home", "other", "reputation"].index(reason)],
        'schoolsup': [0 if schoolsup == "no" else 1],
        'famsup': [0 if famsup == "no" else 1],
        'paid': [0 if paid == "no" else 1],
        'internet': [0 if internet == "no" else 1],
        'age': [age],
        'Medu': [Medu],
        'Fedu': [Fedu],
        'studytime': [studytime],
        'failures': [failures],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'health': [health],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2]
    })

    # Make prediction
    features = ['school', 'age', 'Medu', 'Fedu', 'reason', 'studytime', 'failures', 'schoolsup', 'famsup', 
                'paid', 'internet', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2']
    user_data = user_data[features]
    predicted_grade = model.predict(user_data)

    # Display the predicted grade
    st.write(f"### Predicted Grade: {predicted_grade[0]:.2f}")
    st.balloons()  # Fun animation for prediction

# Add footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
