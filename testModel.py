import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

model = load_model()

# Title
st.title('Flower Type Prediction')

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    size = (150, 150)  # Adjust this size according to your model's input size
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Display predictions
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']  # List of class names
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f'The predicted flower type is: {predicted_class}')
