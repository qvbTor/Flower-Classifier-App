import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

model = load_model()

st.write("# Flower Type Prediction")

file = st.file_uploader("Choose a flower photo from computer", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    flower_types = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_flower = flower_types[np.argmax(prediction)]
    st.success(f"Predicted Flower Type: {predicted_flower}")
