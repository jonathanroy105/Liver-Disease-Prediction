import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('svm_model.pkl')

class_labels = [0,1,2,3,4]

st.title("Liver Disease Prediction App")

st.sidebar.header("Input Features")

def get_user_input():
    age = st.sidebar.number_input("age", min_value=0, max_value=100, step=1)
    sex = st.sidebar.number_input("sex", min_value=0, max_value=1, step=1)
    total_bilirubin = st.sidebar.number_input("bilirubin")
    albumin = st.sidebar.number_input("albumin")
    alkaline_phosphatase = st.sidebar.number_input("alkaline_phosphatase")
    alanine_aminotransferase = st.sidebar.number_input("alanine_aminotransferase")
    aspartate_aminotransferase = st.sidebar.number_input("aspartate_aminotransferase")
    cholesterol = st.sidebar.number_input("cholesterol")
    cholinesterase = st.sidebar.number_input("cholinesterase")
    creatinina = st.sidebar.number_input("creatinina")
    gamma_glutamyl_transferase = st.sidebar.number_input("gamma_glutamyl_transferase")
    protein = st.sidebar.number_input("protein")

    user_data = {
        'age': age,
        'sex': sex,
        'bilirubin': total_bilirubin,
        'albumin': albumin,
        'alkaline_phosphatase': alkaline_phosphatase,
        'alanine_aminotransferase': alanine_aminotransferase,
        'aspartate_aminotransferase': aspartate_aminotransferase,
        'cholesterol': cholesterol,
        'cholinesterase': cholinesterase,
        'creatinina': creatinina,
        'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
        'protein': protein
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

input_data = get_user_input()

st.subheader("User Input:")
st.write(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result:")
    st.write(f"Predicted Class: {class_labels[prediction[0]]}")
