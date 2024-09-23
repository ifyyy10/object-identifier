import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

# Import helper functions from utils.py (if needed)
from utils import color_to_trainId, trainId_to_name

# Load the trained model (ensure the path is correct)
def download_model():
    file_id = '1wUafz9VOoPno0VsrtXMJ8NSzVs_oK_gR'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'cityscape model.h5'
    print(f"Downloading model from {url}...")
    gdown.download(url, output, quiet=False)
    print("Download complete.")

# Check if model file exists, if not, download it
model_file = 'cityscape model.h5'
if not os.path.isfile(model_file):
    download_model()

print("Loading model...")
try:
    model = load_model(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Streamlit App
st.title("Scene Recognition App")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize the image to 256x256
    image = image.resize((256, 256))

    # Preprocess image 
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(img_array)

        # Convert prediction to class trainIds
        predicted_trainId = np.argmax(prediction[0], axis=-1)

        # Get the unique trainIds predicted in the image
        unique_trainIds = np.unique(predicted_trainId)

        # Map trainIds to class names
        predicted_classes = [trainId_to_name[trainId] for trainId in unique_trainIds if trainId in trainId_to_name]

        # Display the predicted classes
        st.write("Predicted Classes in the Image:")
        st.write(predicted_classes)

        # Convert prediction to RGB image for visualization
        predicted_rgb = np.zeros((prediction.shape[1], prediction.shape[2], 3))
        for color, trainId in color_to_trainId.items():
            mask = predicted_trainId == trainId
            predicted_rgb[mask] = color

        # Create and display the predicted segmentation image
        predicted_image = Image.fromarray(predicted_rgb.astype('uint8'))
        st.image(predicted_image, caption='Predicted Segmentation', use_column_width=True)
