import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

# Load the model
model = load_model()

# Title
st.write("""
# Flower Type Prediction
""")

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    try:
        size = (150, 150)
        image = image_data.resize(size)  # Resize without ANTIALIAS
        img = np.asarray(image)
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display predictions
if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        if prediction is not None:
            class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            st.success(f'Predicted flower type: {predicted_class}')
    except Exception as e:
        st.error(f"Error processing image: {e}")
