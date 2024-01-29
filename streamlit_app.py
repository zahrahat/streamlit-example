import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the map HTML
def load_map():
    with open('real_estate_properties_map.html', 'r') as f:
        map_html = f.read()
    return map_html

# Function to load the model
def load_model():
    model = tf.keras.models.load_model('real_estate_price_prediction_model.h5')
    return model

# Function to preprocess input data
def preprocess_input(surface, pieces, sdb, chambres, age, etage):
    input_data = np.array([[surface, pieces, sdb, chambres, age, etage]])
    return input_data

# Function to make predictions
def predict_price(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0][0]
    return prediction

# Load the map HTML
map_html = load_map()

# Load the model
model = load_model()

# Load cleaned data
df = pd.read_excel('your_data_file.xlsx')  # Update with your data file path

# Define features and target variable
excluded_features = ['name', 'price']
X = df.drop(columns=excluded_features).values
y = df['price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling or Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Main function to run the app
def main():
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
        input_data = preprocess_input(surface, pieces, sdb, chambres, age, etage)
        prediction = predict_price(model, scaler, input_data)
        st.write(f'Predicted Price: {prediction}')

    # Display the map HTML
    st.header('Real Estate Properties Map')
    st.markdown('Map showing real estate properties')
    st.components.v1.html(map_html, height=600)

    # Additional analyses
    st.header('Additional Analyses')
    st.markdown('Additional visualizations and analyses go here')
    # Overview
    st.write(df['price'].describe())

    # Distribution of Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Real Estate Prices')
    plt.xlabel('Price (Currency)')
    plt.ylabel('Frequency')
    st.pyplot()

    # Categorical Features Analysis
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='surface', y='price', data=df, palette='viridis')
    plt.title('Property Condition vs. Price')
    plt.xlabel('Condition')
    plt.ylabel('Price')
    st.pyplot()

    # Numerical Features Analysis
    sns.pairplot(df[['price', 'surface', 'pieces', 'sdb', 'chambres']])
    st.pyplot()

    # Price Distribution by Currency
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='pieces', y='price', data=df, palette='muted')
    plt.title('Price Distribution by Currency')
    plt.xlabel('Currency')
    plt.ylabel('Price')
    st.pyplot()

if __name__ == '__main__':
    main()
