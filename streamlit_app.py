import streamlit as st
from PIL import Image
import os
import shutil
from keras.models import load_model
from helpers import load_labels, load_image, preprocess_image, show_labels

@st.cache_resource
def getModel():
    # This function will only be run the first time it's called
    loaded_model = load_model('saved_models/vgg16ft_2_finetune.h5')
    print("************** Model Loaded **************")
    return loaded_model

@st.cache_data
def getLabels():
    loaded_labels = load_labels('labels.txt')
    loaded_labels.sort()
    print("************** Labels Loaded **************")
    return loaded_labels

if __name__ == '__main__':
    # Load Whole model
    model = getModel()

    # Load Labels
    labels = getLabels()

    st.title("Animal-RoadKill-Prevention Task")
    st.caption("A fine-tuned 47 class animal VGG16 classifier by Karim Ibrahim")

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
        img = load_image(image_path)

        # Pre process data
        img = preprocess_image(img)

        # Perform inference
        pred = model.predict(img)

        # Removing the "uploads" folder
        shutil.rmtree("uploads")

        # Show the user that we have finished
        onscreen.empty()

        st.subheader("Predicted Label: \n"+show_labels(pred, labels))
        # onscreen.text()
