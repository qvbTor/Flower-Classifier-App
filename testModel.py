import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('xx.h5')
    return model

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize the image to match model's expected sizing
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def make_prediction(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Display the prediction result
def display_prediction(predictions):
    class_labels = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Main function to run the Streamlit app
def main():
    # Load the model
    model = load_model()

    # Set up the Streamlit app layout
    st.title("Flower Classifier")
    st.write("Upload an image of a flower to classify its species.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make predictions
        predictions = make_prediction(model, img_array)

        # Display the prediction result
        predicted_class = display_prediction(predictions)
        st.success(f"The predicted flower species is: {predicted_class}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
