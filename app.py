import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')
# Sidebar setup
st.sidebar.image('Ahmad Ali.png', use_column_width=True)
st.sidebar.header("**Ahmad Ali Rafique**")
st.sidebar.write("AI & Machine Learning Expert")

st.sidebar.header("Contact Information", divider='rainbow')
st.sidebar.write("Feel free to reach out through the following")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)")
st.sidebar.write("[GitHub](https://github.com/Ahmad-Ali-Rafique/)")
st.sidebar.write("[Email](mailto:arsbussiness786@gmail.com)")
st.sidebar.write("Developed by Ahmad Ali Rafique", unsafe_allow_html=True)

# Title of the application
st.title("USA House Price Prediction")

# Description of the application
st.write("""
This application predicts the price of a house in the USA based on several input features.
Please enter the details below to get an estimated price for a house.
""")

# Input fields for user input
sqft_living = st.number_input('Square Footage of Living Area (sqft)')
bedrooms = st.number_input('Number of Bedrooms')
bathrooms = st.number_input('Number of Bathrooms')
floors = st.number_input('Number of Floors')
zipcode = st.number_input('Zip Code')

# Prepare the input for prediction
input_data = np.array([[sqft_living, bedrooms, bathrooms, floors, zipcode]])

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f"The estimated house price is: ${prediction[0]:,.2f}")

# Footer or additional information
st.write("""
*Note: This prediction is based on the model trained with historical data and may not reflect the current market conditions.*
""")

