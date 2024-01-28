import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title('Simple Streamlit App')

# User input
x = st.slider('Select a value', 0.0, 10.0, 5.0)

# Generate data
y = np.sin(x)

# Plot data with specified backend
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
