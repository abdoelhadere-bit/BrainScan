import os
import json

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="BrainScan AI", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model_and_classes():
    keras_path = os.path.join(MODEL_DIR, "brainscan_model.keras")
    h5_path = os.path.join(MODEL_DIR, "brainscan_model.h5")
    model = tf.keras.models.load_model(keras_path if os.path.exists(keras_path) else h5_path)
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(file_bytes):
    """Même prétraitement que l'entraînement : gris, 224x224, [0, 1]."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype("float32") / 255.0
    return img, x[np.newaxis, ..., np.newaxis]


st.title("BrainScan AI")
st.caption("Classification d'IRM cérébrales : glioma, meningioma, pituitary, notumor")

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Choisir une IRM", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    img, x = preprocess(uploaded.read())
    if img is None:
        st.error("Image illisible.")
    else:
        col1, col2 = st.columns(2)
        col1.image(img, caption="IRM analysée", use_container_width=True, clamp=True)

        proba = model.predict(x, verbose=0)[0]
        k = int(np.argmax(proba))

        col2.metric("Prédiction", class_names[k], f"{proba[k]:.1%} de confiance")
        col2.bar_chart({c: float(p) for c, p in zip(class_names, proba)})

        if class_names[k] != "notumor":
            st.warning("Une tumeur est suspectée. Ce résultat doit être confirmé par un médecin.")

st.divider()
st.caption("Outil d'aide au diagnostic uniquement. Ne remplace pas l'avis d'un professionnel de santé.")