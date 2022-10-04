import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def welcome():
    st.markdown("<h1 style='padding:2rem;text-align:center; background-color:white;color:black;font-size:1.8rem;border-radius:0.8rem;'>Titanic Shipwreck Survival Prediction</h1>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<p style='font-famly:Arial, Helvetica, sans-serif; color:black;'>The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This app is build to answer the question: “what sorts of people were more likely to survive?” using passenger data (i.e., name, age, gender, socio-economic class, etc).</p1>", unsafe_allow_html=True)
    st.write("---")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-repeat: no-repeat;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    
def apk():
    model = pickle.load(open("survival_classifier.pkl", 'rb'))
    st.markdown("<h3 style='padding:2rem;text-align:left;color:black;font-size:1.8rem;border-radius:0.5rem;'>Survival Prediction</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-famly:Arial, Helvetica, sans-serif; color:black;'>By entering input parameters, the model outputs predicted Survival.</p1>", unsafe_allow_html=True)
    data = pd.DataFrame()

    # uploaded = st.file_uploader("Choose file")
    
    Sex = 0
    Sex = st.radio(
    "Sex",
    ('Female', 'Male'))

    if Sex == 'Female':
        sex == 0
    elif Sex == 'Male':
        sex == 1

    Parch = 0
    Parch = st.radio(
    "Number of Parent/children",
    (0, 1, 2, 3, 4, 5, 6, 9))
    
    p = 0
    pclass = st.radio("Passenger Class",
                      ("First", "Second", "Third"))
    if pclass == 'First':
        p == 1
    
    elif pclass == 'Second':
        p == 2
    
    elif pclass == 'Third':
        p == 3
    
    age = st.text_input("Age")

    if age:
        try:
            age=int(age)
            if(age>90 or age<0):
                st.write("Age should be between 0 and 90")
        except Exception as e:
            st.write(e)
    
    sib = 0
    sib = st.radio(
    "Number of siblings",
    (0, 1, 2, 3, 4, 5, 8))
    
    embark=0
    Embark=st.radio("Embark", ("Cherbourg", "Queenstown", "Southampton"))
    if Embark=='Cherbourg':
        embark==0
        
    elif Embark=='Queenstown':
        embark==1
    
    elif Embark=='Southampton':
        embark==2
    

    fare = st.text_input("Fare")

    if fare:
        try:
            fare=float(fare)
        except:
            st.write("Please enter a number.")
            
    id_ = st.text_input("Passenger ID")
    
    if id_:
        try:
            id_=int(id_)
        except:
            st.write("Please enter a number.")
    
    
    calculate = st.button('Predict')

        
    if calculate:
        try:
            input_= [[id_, p, sex, age, sib, Parch, fare, embark]]
            predict = model.predict(input_)
            if predict==0:
                st.markdown("<h3 style='padding:2rem;text-align:left;color:red;font-size:1.8rem;border-radius:0.5rem;'>Died!</h3>", unsafe_allow_html=True) 

            if predict==1:
                st.markdown("<h3 style='padding:2rem;text-align:left;color:green;font-size:1.8rem;border-radius:0.5rem;'>Survived!</h3>", unsafe_allow_html=True) 
        
        except:
            st.write("please enter all necessary information")


welcome()
add_bg_from_local('Titanicsetsoff.jpg')
apk()
