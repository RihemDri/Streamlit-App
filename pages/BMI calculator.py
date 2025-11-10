#import module
import streamlit as st
import pandas as pd
import numpy as np

st.header("BMI Calculator")
st.write("Calculate your Body Mass Index (BMI) using this simple calculator.")
# Input fields for weight and height
weight = st.number_input("Enter your weight in kilograms (kg):")
radio_unit= st.radio("Select your height unit:",("Centimeters","Meters","Feet"))

if radio_unit=="Centimeters":
   height = st.number_input('enter your height in  Centimeters', min_value=0.0)
   if height > 0:
    bmi = weight / ((height / 100) ** 2)
elif radio_unit=="Meters":
   height = st.number_input('enter your height in Meters', min_value=0.0)
   if height > 0:
    bmi = weight / (height ** 2)
else:
   height = st.number_input('enter your height in Feet', min_value=0.0)
   if height > 0:
    bmi = weight / ((height * 0.3048) ** 2)


# Calculate BMI when the button is clicked
if st.button("Calculate BMI"):
 if bmi is not None:
    st.text(f"Your BMI Index is {bmi:.2f}")

    if bmi < 16:
      st.error("Extremely Underweight")
    elif bmi < 18.5:
      st.warning("Underweight")
    elif bmi < 25:
      st.success("Healthy")
    elif bmi < 30:
      st.warning("Overweight")
    else:
      st.error("Extremely Overweight")
 else:
    st.warning("Please enter valid weight and height before calculating BMI.")