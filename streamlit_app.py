import shutil

import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from helpers import load_labels, show_labels
import numpy as np
@st.cache_resource
def getModel():
    # This function will only be run the first time it's called
    loaded_model = EfficientNetB0(include_top=True, weights='imagenet')
    print("************** Model Loaded **************")
    return loaded_model

@st.cache_data
def getLabels():
    # Loading Labels
    loaded_labels = load_labels('labels_2.txt')
    print("************** Labels Loaded **************")
    return loaded_labels

if __name__ == '__main__':
    # Load Whole model
    model = getModel()

    # Load Labels
    labels = getLabels()

    st.title("Animal-RoadKill-Prevention Task")
    st.caption("An animal classifier for Elevate AI by Karim Ibrahim")

    st.subheader("Image Uploader:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Displaying the image
        st.subheader("Uploaded Image")
        st.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)

        # Give the user an on-screen indication that we are working
        onscreen = st.empty()
        onscreen.text('Classifying...')

        # Creating a folder and adding the image to it. Then, getting the file path
        if not os.path.isdir("uploads"):
            os.mkdir("uploads")
        image_path = "uploads/"+uploaded_file.name+""
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load input data
        img = image.load_img(image_path, target_size=(224, 224))

        # Convert the image to a NumPy array
        x = image.img_to_array(img)

        # Expand the dimensions of the array to create a batch of size 1
        x = tf.expand_dims(x, axis=0)

        # Pre process data
        x = preprocess_input(x)

        # Perform inference
        pred = model.predict(x)

        # Removing the "uploads" folder
        shutil.rmtree("uploads")

        # Show the user that we have finished
        onscreen.empty()

        st.subheader("Predicted Label: \n"+show_labels(pred, labels))
        # onscreen.text()
