import os
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf

def load_model_or_prompt(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            st.error(f"Failed to load model at {path}: {e}")
    st.error("Pretrained model not found. Please add `model/digit_recognition_model.keras` to the repository.\n\nSee README for Colab training and upload instructions.")
    st.stop()

def preprocess_pil_image(pil_img):
    img = pil_img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img).astype('float32')/255.0
    arr = arr.reshape(1,28,28,1)
    return arr, img
