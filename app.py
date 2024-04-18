import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

st.set_page_config(
  page_title="Animal Image Detection App",
  page_icon="📱",
)

st.sidebar.success(Select a page above👆")

st.title("Animal Image Detection Application")
st.header("Image Classifier ")

model = tf.keras.models.load_model("Animal_Classification.h5")
### load file
uploaded_file = st.file_uploader("Upload an Image for Image classification", type=["jpg","jpeg","png"])

map_dict = { 0: 'butterfly',
             1: 'cat',
             2: 'chicken',
             3: 'cow',
             4: 'dog' ,
             5: 'elephant',
             6: 'horse',
             7: 'sheep',
             8: 'spider',
             9: 'squirrel'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
