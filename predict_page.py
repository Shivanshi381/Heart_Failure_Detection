#Predict_page
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




def load_model():
    with open ("Saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rfc = data["model"]
chest_pain_type = ["Typical Angina","Atypical Angina","Non-Anginal Pain", "Asymptomatic"]
sc = StandardScaler()
def show_predict_page():
    st.title("Heart Failure Detection ")
    st.write("""We need clinical data for the diagnosis""")
    age = st.slider("Age", min_value=1,max_value=115)
    sex = st.selectbox("Sex", ["Male", "Female"])

    

    chest_pain = st.selectbox("Chest pain type",chest_pain_type)
    

    bloodpressure = st.slider("Resting blood pressure (in mm Hg onadmission to hospital)",min_value=10,max_value=250)

    cholesterol = st.slider("Serum Cholesterol in mg/dl",min_value=100,max_value=450)

    fasting_bloodsugar = st.selectbox("Is the fasting bloodsugar > 120 mg/dl",["Yes", "No"])


    ecg = st.selectbox("ECG",["Normal", "Having ST-T wave abnormality", "Left ventricular hypertrophy"])


    max_heart_rate = st.slider("Maximum heart rate",min_value=50,max_value=250)

    exercise_angina = st.selectbox("Exercise induced angina",["Yes", "No"])


    old_peak = st.selectbox("Old peak (ST slope induced by exercise relative to rest)",[0,1,1.5,2])

    st_slope = st.selectbox("ST Slope (Slope of the peak exercise ST11 Slope segment)",["Up slopping", "Flat", "Down slopping"])


    

    ok = st.button("Calculate the risk")
    
    if ok:
        X = np.array([[age,sex,chest_pain,bloodpressure,cholesterol,fasting_bloodsugar,ecg,max_heart_rate,exercise_angina,old_peak,st_slope]])
        st.write(X)
        if sex == "Male":
            sex =1 
        if sex == "Female":
            sex =0
        if chest_pain == "Typical Angina":
            chest_pain =1
        if chest_pain == "Atypical Angina":
            chest_pain =2
        if chest_pain == "Non-Anginal Pain":
            chest_pain =3
        if chest_pain == "Asymptomatic":
            chest_pain =4
        if fasting_bloodsugar== "Yes":
            fasting_bloodsugar =1
        if fasting_bloodsugar == "No":
            fasting_bloodsugar=0
        if ecg == "Normal":
            ecg = 0
        if ecg == "Having ST-T wave abnormality":
            ecg = 1
        if ecg == "Left ventricular hypertrophy":
            ecg = 2
        if exercise_angina == "Yes":
            exercise_angina=1
    
        if exercise_angina == "No":
            exercise_angina=0
        if st_slope == "Up slopping":
            st_slope =1
        if st_slope == "Flat":
            st_slope =2
        if st_slope == "Down slopping":
            st_slope =3

        X = np.array([[age,sex,chest_pain,bloodpressure,cholesterol,fasting_bloodsugar,ecg,max_heart_rate,exercise_angina,old_peak,st_slope]])
        prediction = rfc.predict(X)
        if prediction == 0:
            st.subheader("No major risk found")
        else:
            st.subheader("High risk of heart disease.")



    



