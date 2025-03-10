import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "Model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "Model", "class_indices.json")

gdrive_url = "https://drive.google.com/uc?id=161k3OqiMJjAP7__6x5eXzropqGCzlESs"

if not os.path.exists(model_path):
    st.info("Downloading model file...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(gdrive_url, model_path, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load class indices
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
index_to_class = {int(k): v for k, v in class_indices.items()}


def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array.astype('float32') / 255.0 
    return img_array

def predict_disease(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = index_to_class[predicted_index]
    return predicted_class

st.title("🌱 Plant Disease Prediction")
st.write("Upload an image of a plant leaf to detect possible diseases.")

uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_width = 400  # Adjust the width as needed
        image.thumbnail((max_width, max_width))  # Resize the image
        st.image(image, caption="Uploaded Image", use_container_width=False)


    with col2:
        if st.button("🔍 Predict"):
            prediction = predict_disease(image)
            st.success(f"**Prediction: {prediction}**")
