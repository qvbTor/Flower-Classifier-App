import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('flower_classification_model.h5')
    return model

model = load_model()

# Title
st.title('Flower Type Prediction')

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    try:
        size = (150, 150)
        image = ImageOps.fit(image_data, size)
        img_array = np.asarray(image)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Display predictions
if file is not None:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        if prediction is not None:
            st.success(f'The predicted flower type is: {prediction}')
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.text("Please upload an image file")
