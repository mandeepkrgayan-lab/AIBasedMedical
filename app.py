import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


model = load_model("xray_pneumonia_model.h5")

st.title("Chest X-ray Pneumonia Detector")
st.write("Upload a chest X-ray image and let the AI detect pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)

    
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

   
    prediction = model.predict(img_array)[0][0]

    label = "- Normal" if prediction < 0.5 else "+ Pneumonia Detected"
    confidence = (1 - prediction) if prediction < 0.5 else prediction

    st.subheader(label)
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
