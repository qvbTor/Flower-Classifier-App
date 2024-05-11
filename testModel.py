import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

model = load_model()
st.write("""
# Flower Type Prediction
""")
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = Image.open(image_data)
    resized_image = image.resize(size)
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    st.image(file, use_column_width=True)
    prediction = import_and_predict(file, model)
    class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    st.success(f'The predicted flower type is: {predicted_class}')
