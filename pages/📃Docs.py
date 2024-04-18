import streamlit as st

st.header("Documentation:"),

st.subheader("Abstract");

st.write("The project aims to develop an interactive web application for animal image classification using Streamlit, a Python library for building web applications. The application allows users to upload an image of an animal, which is then processed by a pre-trained convolutional neural network (CNN) model to predict the animal's class.The application's backend utilizes TensorFlow for image preprocessing and prediction. Uploaded images are resized, normalized, and fed into the CNN model to obtain a prediction. The predicted animal class is displayed to the user along with the uploaded image.The frontend is developed using Streamlit, providing a simple and intuitive interface for users to interact with. Users can easily upload an image, view the prediction, and receive real-time feedback on the predicted animal class.Overall, this project demonstrates the integration of deep learning models with web technologies using Streamlit, enabling rapid prototyping and deployment of machine learning applications accessible through a web browser. The resulting application serves as a practical tool for animal image classification and can be extended with additional features and enhancements for broader use cases.")
