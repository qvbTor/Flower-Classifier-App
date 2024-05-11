import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

st.write("""
# SHEESH
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def preprocess_image(image_data):
    size = (64, 64)
    image = Image.open(image_data)
    image = image.resize(size)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = img_array[np.newaxis, ...]
    return img_array

def import_and_predict(image_data, model):
    img_array = preprocess_image(image_data)
    st.write("Image shape:", img_array.shape)  # Debugging
    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)  # Debugging
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = load_model()
    prediction = import_and_predict(file, model)
    st.write("Prediction shape:", prediction.shape)  # Debugging
