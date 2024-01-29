import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('real_estate_price_prediction_model.h5')

# Define the user interface
st.title('Real Estate Price Prediction')

# Input fields for user to provide data
surface = st.number_input('Surface Area', min_value=0.0, value=100.0)
pieces = st.number_input('Number of Rooms', min_value=0, value=3)
sdb = st.number_input('Number of Bathrooms', min_value=0, value=2)
chambres = st.number_input('Number of Bedrooms', min_value=0, value=2)
age = st.number_input('Age of Property', min_value=0.0, value=5.0)
etage = st.number_input('Floor', min_value=0, value=1)

# Button to trigger prediction
if st.button('Predict Price'):
    # Preprocess input data
    input_data = np.array([[surface, pieces, sdb, chambres, age, etage]])
    # Scale input data (if necessary)
    # Make prediction
    prediction = model.predict(input_data)[0][0]
    # Display prediction
    st.write(f'Predicted Price: {prediction}') 
