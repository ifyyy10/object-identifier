import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

# Import helper functions
from utils import color_to_trainId, trainId_to_name

# Download model if not present
def download_model():
    file_id = '1wUafz9VOoPno0VsrtXMJ8NSzVs_oK_gR'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'cityscape_model.h5'
    gdown.download(url, output, quiet=False)

model_file = 'cityscape_model.h5'
if not os.path.isfile(model_file):
    download_model()

model = load_model(model_file)

# Streamlit App
st.title("Scene Recognition App")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize the image to 256x256
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('Predict'):
        prediction = model.predict(img_array)
        predicted_trainId = np.argmax(prediction[0], axis=-1)

        # Unique trainIds in the image
        unique_trainIds = np.unique(predicted_trainId)

        # Create RGB segmentation image
        predicted_rgb = np.zeros((prediction.shape[1], prediction.shape[2], 3))
        for color, trainId in color_to_trainId.items():
            mask = predicted_trainId == trainId
            predicted_rgb[mask] = color

        predicted_image = Image.fromarray(predicted_rgb.astype('uint8'))

        # Add labels on the output image
        draw = ImageDraw.Draw(predicted_image)
        font = ImageFont.load_default()  # For simplicity; add a custom font path if needed
        for trainId in unique_trainIds:
            if trainId in trainId_to_name:
                class_name = trainId_to_name[trainId]
                # Add text labels at random positions (or calculate better positions)
                pos = np.argwhere(predicted_trainId == trainId)
                if pos.size > 0:
                    y, x = pos[0]
                    draw.text((x, y), class_name, fill=(255, 255, 255), font=font)

        # Display the labeled segmentation image
        st.image(predicted_image, caption='Labeled Segmentation Image', use_column_width=True)
