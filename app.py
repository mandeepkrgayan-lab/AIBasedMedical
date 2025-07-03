import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

@st.cache_resource
def load_model_cached():
    if not os.path.exists("xray_model.h5"):
        gdown.download("https://drive.google.com/uc?id=1CVAU8cY-V945wTWe-L1mn77_Bhn2dxR-", "xray_pnuemonia_model.h5", quiet=False)
    return load_model("xray_pnuemonia_model.h5")

model = load_model_cached()


st.title("🩻 Chest X-ray Pneumonia Detector")
st.write("Upload a chest X-ray image and let the AI detect pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)

   
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Model expects 4D input

   
    prediction = model.predict(img_array)[0][0]

   
    label = "🟢 Normal" if prediction < 0.5 else "🔴 Pneumonia Detected"
    confidence = (1 - prediction) if prediction < 0.5 else prediction

    st.subheader(label)
    st.write(f"Confidence: **{confidence * 100:.2f}%**")