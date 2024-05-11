import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

st.write("""
# SHEESH
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

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
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = load_model()
    prediction = import_and_predict(file, model)
    class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = np.argmax(prediction)
    string = "OUTPUT : " + class_names[predicted_class_index]
    st.success(string)
