import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = load_model('traffic_sign_model.h5')

st.title("🚦 Traffic Sign Recognition System")

uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(64, 64))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    st.success(f"Predicted Sign Class: **{predicted_class}**")
