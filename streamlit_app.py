import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('real_estate_price_prediction_model.h5')

# Function to preprocess input data
def preprocess_input(surface, pieces, sdb, chambres, age, etage):
    # Example preprocessing steps (adjust as needed)
    input_data = np.array([[surface, pieces, sdb, chambres, age, etage]])
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled

# Define the user interface
st.title('Moroccan Real Estate Price Prediction')

# Load the map
st.markdown('<h2 style="text-align:center;">Real Estate Properties Map</h2>', unsafe_allow_html=True)
st.markdown('<iframe src="real_estate_properties_map (1).html" width="100%" height="500"></iframe>', unsafe_allow_html=True)

# Input fields for user to provide data
st.sidebar.title('Enter Property Details')
surface = st.sidebar.number_input('Surface Area', min_value=0.0, value=100.0)
pieces = st.sidebar.number_input('Number of Rooms', min_value=0, value=3)
sdb = st.sidebar.number_input('Number of Bathrooms', min_value=0, value=2)
chambres = st.sidebar.number_input('Number of Bedrooms', min_value=0, value=2)
age = st.sidebar.number_input('Age of Property', min_value=0.0, value=5.0)
etage = st.sidebar.number_input('Floor', min_value=0, value=1)

# Button to trigger prediction
if st.sidebar.button('Predict Price'):
    # Preprocess input data
    input_data_scaled = preprocess_input(surface, pieces, sdb, chambres, age, etage)
    # Make prediction
    prediction = model.predict(input_data_scaled)[0][0]
    # Display prediction
    st.sidebar.write(f'Predicted Price: {prediction}')
