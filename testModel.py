import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

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
    image = np.array(Image.open(image_data))
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(resized_image, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    st.image(file, use_column_width=True)
    prediction = import_and_predict(file, model)
    class_labels = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    st.success(f'The predicted flower type is: {predicted_class}')
