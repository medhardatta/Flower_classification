import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Flower Classifier")
st.write("Upload a flower image and the model will predict its type.")

# Model is hosted on Hugging Face Hub (too large for GitHub's 100MB limit).
# CHANGE THIS to your own Hugging Face repo id once you've uploaded the model.
HF_REPO_ID = "medhardatta/flower_classifier"
HF_FILENAME = "best_flower_model.keras"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Order confirmed from training notebook output (alphabetical, from image_dataset_from_directory)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Match training preprocessing EXACTLY: resize to 224x224, raw pixel values (0-255),
    # NO division by 255 — the model has no Rescaling layer.
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    st.success(f"**Prediction:** {predicted_class.capitalize()} ({confidence:.1f}% confidence)")

    with st.expander("See all class probabilities"):
        for name, prob in sorted(zip(class_names, prediction[0]), key=lambda x: -x[1]):
            st.write(f"{name}: {prob*100:.2f}%")
