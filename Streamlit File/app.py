import streamlit as st
import numpy as np
from PIL import Image
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Page config
st.set_page_config(
    page_title="Wildfire Detection",
    page_icon="🔥",
    layout="centered"
)

MODEL_PATH = r"D:\desktop\Gazanfar Ali - DL Project\mobile_model.h5"

@st.cache_resource
def load_model():
    model = keras.saving.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# UI
st.title("🔥 Wildfire Detection")
st.write("Upload an image to detect if there is a wildfire.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]

    st.markdown("---")
    if prediction > 0.5:
        st.error(f"🔥 **FIRE DETECTED** — Confidence: {prediction * 100:.1f}%")
    else:
        st.success(f"✅ **NO FIRE** — Confidence: {(1 - prediction) * 100:.1f}%")

    st.progress(float(prediction))
    st.caption(f"Raw prediction score: {prediction:.4f}")
