import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

model = load_model()

# Title
st.write("""
# Flower Type Prediction
""")

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    size = (150, 150)
    image = Image.open(image_data)
    image = image.resize(size)
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Display predictions
if file is not None:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(file, model)
        class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']  # Assuming these are your class labels
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]
        st.success(f'The predicted flower type is: {predicted_class}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.text("Please upload an image file")
