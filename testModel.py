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
    img = image.resize((150, 150))  
    img_array = np.array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
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

# Flower descriptions
flower_descriptions = {
    'Lilly': "Lilies are known for their large, prominent flowers and their pleasant fragrance. They are commonly used in floral arrangements and symbolize purity and renewal.",
    'Lotus': "The lotus flower holds great significance in many cultures and religions, often representing purity, enlightenment, and rebirth. It is also known for its beauty and resilience.",
    'Orchid': "Orchids are prized for their exquisite beauty and diversity. They come in various colors, shapes, and sizes, and are often associated with love, luxury, and refinement.",
    'Sunflower': "Sunflowers are known for their bright yellow petals and large, round flower heads. They symbolize adoration, loyalty, and longevity, and are often associated with warmth and happiness.",
    'Tulip': "Tulips are popular spring flowers known for their vibrant colors and elegant appearance. They symbolize perfect love, beauty, and prosperity."
}

# Main function to run the Streamlit app
def main():
    # Load the model
    model = load_model()

    # Set the theme to dark
    st.markdown("""
        <style>
            .reportview-container {
                background: #1E1E1E;
                color: white;
            }
            .sidebar .sidebar-content {
                background: #1E1E1E;
            }
        </style>
    """, unsafe_allow_html=True)

    # Set up the Streamlit app layout
    st.title("Flower Classifier")
    st.markdown("This app uses a pre-trained model to classify images of flowers into one of the following categories: Lilly, Lotus, Orchid, Sunflower, or Tulip.")

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
        if predicted_class in flower_descriptions:
            st.success(f"The predicted flower species is: {predicted_class}")
            st.write(f"Description: {flower_descriptions[predicted_class]}")
        else:
            st.error("Unrecognized flower or object.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
