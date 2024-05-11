import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("xx.h5")

# Load and preprocess the test image
img_path = "random test/6.jpg"  # Replace "path_to_test_image.jpg" with the path to your test image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make predictions
predictions = model.predict(img_array)

# Get the predicted class label
predicted_class_index = np.argmax(predictions[0])
class_labels = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']  # Assuming these are your class labels
predicted_class = class_labels[predicted_class_index]

# Evaluate the model on the test image
accuracy = model.evaluate(img_array, np.array([[1 if i == predicted_class_index else 0 for i in range(len(class_labels))]]))[1]

print("Predicted class:", predicted_class)
print("Model Accuracy:", accuracy)
