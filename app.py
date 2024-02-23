import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model(r'D:\CT_kidney\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\kidney-4-(224 X 250)-68.47.h5')  # Replace 'your_model.h5' with the path to your model file

# Define class labels
class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size expected by your model
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Normalize the pixel values (if needed)
    # Perform any other preprocessing steps required by your model
    return image_array

# Define the Streamlit UI
st.title('Kidney CT Scan Image Classifier')

uploaded_file = st.file_uploader("Upload a kidney CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make predictions
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = class_labels[np.argmax(prediction)]
    
    st.write('Predicted Class:', predicted_class)
