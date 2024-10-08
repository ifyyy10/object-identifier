import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Import helper functions from utils.py (if needed)
from utils import color_to_trainId, trainId_to_name 

# Load the trained model (ensure the path is correct)
# Function to download model from Google Drive
def download_model():
    file_id = '1xJBec2aPibWKWYMfoOz2K5hiq16QeI0x'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'cityscapes model.h5'
    print(f"Downloading model from {url}...")
    gdown.download(url, output, quiet=False)
    print("Download complete.")

# Check if model file exists, if not, download it
model = 'cityscapes model.h5'
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
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image 
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(img_array)

        # Convert prediction to RGB image (use your existing logic)
        predicted_rgb = np.zeros((prediction.shape[1], prediction.shape[2], 3))
        for color, trainId in color_to_trainId.items():
            mask = np.argmax(prediction[0], axis=-1) == trainId
            predicted_rgb[mask] = color

        predicted_image = Image.fromarray(predicted_rgb.astype('uint8'))

        # Annotate the predicted image (optional, use your existing logic)
        # ... 

        # Display the result
        st.image(predicted_image, caption='Predicted Segmentation', use_column_width=True)
