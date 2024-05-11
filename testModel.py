import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("xx.h5")
    return model

model = load_model()

# Define class labels
class_labels = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    try:
        size = (150, 150)
        # Resize the image and convert it to grayscale
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS).convert("RGB")
        img_array = np.asarray(image)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # Make predictions
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit UI
st.title("Flower Type Prediction")

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        # Display the uploaded image
        image = Image.open(file)
        st.image(image, use_column_width=True)
        
        # Make prediction
        prediction = import_and_predict(image, model)
        
        if prediction is not None:
            # Get the predicted class
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]
            st.success(f"The predicted flower type is: {predicted_class}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
