import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("xx.h5")
    return model

model = load_model()

# Define class labels
class_labels = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    img = image.load_img(image_data, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Function to make predictions
def predict(image_data, model):
    img_array = preprocess_image(image_data)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    accuracy = predictions[0][predicted_class_index] * 100  # Confidence score
    return predicted_class, accuracy

# Streamlit app
st.title("Flower Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    predicted_class, confidence = predict(uploaded_file, model)

    # Display predictions
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
