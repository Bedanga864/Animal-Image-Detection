import streamlit as st

st.set_page_config(
  page_title="Animal Image Detection App",
  page_icon="📱",
)

st.subheader('Docementation', divider='rainbow')
st.subheader('Abstract')
st.write("The project aims to develop an interactive web application for animal image classification using Streamlit, a Python library for building web applications. The application allows users to upload an image of an animal, which is then processed by a pre-trained convolutional neural network (CNN) model to predict the animal's class.The application's backend utilizes TensorFlow for image preprocessing and prediction. Uploaded images are resized, normalized, and fed into the CNN model to obtain a prediction. The predicted animal class is displayed to the user along with the uploaded image.The frontend is developed using Streamlit, providing a simple and intuitive interface for users to interact with. Users can easily upload an image, view the prediction, and receive real-time feedback on the predicted animal class.Overall, this project demonstrates the integration of deep learning models with web technologies using Streamlit, enabling rapid prototyping and deployment of machine learning applications accessible through a web browser. The resulting application serves as a practical tool for animal image classification and can be extended with additional features and enhancements for broader use cases.")

st.subheader("Introduction")
st.write("In today's era of machine learning and web technology integration, the development of interactive applications for image classification has become increasingly accessible and impactful. This project explores the creation of an animal image classifier using Streamlit, a Python library designed for building web applications around data science and machine learning models.The objective is to develop a user-friendly web interface that allows individuals to upload images of animals and receive real-time predictions about the animal's species. Leveraging the power of transfer learning and pre-trained convolutional neural networks (CNNs), the model behind the application has been fine-tuned to recognize various animal classes.This project showcases the fusion of deep learning with web development, enabling a seamless user experience where complex machine learning tasks are encapsulated within a straightforward and accessible web application. ")

st.subheader("Methodology")
st.write("The development of the animal image classification project involves several key steps, including model preparation, Streamlit app creation, and deployment. Below is a detailed methodology outlining each stage of the project:")

st.write("Data Collection and Preparation:")
st.markdown(" # -Gather a diverse dataset of animal images containing different species such as cats, dogs, birds, and horses.")
st.markdown(" # -Organize the dataset into training, validation, and test sets.")
st.write("Model Selection and Transfer Learning:")
st.markdown("-Choose a pre-trained CNN model as the base architecture (e.g., VGG, ResNet, MobileNet) due to their effectiveness in image classification tasks.")
st.markdown("-Fine-tune the selected model on the collected animal dataset to adapt it to recognize specific animal classes.")
st.write("Training the Model:")
st.markdown("-Use transfer learning to train the modified CNN model on the animal dataset.")
st.markdown("-Utilize techniques like data augmentation to improve model generalization and performance.")
st.write("Model Evaluation:")
st.markdown("-Evaluate the trained model using the validation set to assess its accuracy and performance metrics.")
st.write("Model Serialization and Deployment:")
st.markdown("-Serialize the trained model to a format compatible with TensorFlow serving.")
st.markdown("-Prepare the model for integration into the Streamlit application.")
st.write("Streamlit Application Development:")
st.markdown("-Set up a Python environment with necessary libraries including Streamlit, TensorFlow, PIL (Pillow), and NumPy.")
st.write("Develop a Streamlit web application (app.py) that includes:")
st.markdown("-File uploader for users to upload animal images.")
st.markdown("-Preprocessing function to resize and normalize uploaded images.")
st.markdown("-Integration of the trained model to perform predictions on uploaded images.")
st.markdown("-Display of prediction results to users along with the uploaded image.")
st.write("Local Testing and Debugging:")
st.markdown("-Test the Streamlit application locally to ensure it functions correctly.")
st.markdown("-Debug any issues related to model integration, image preprocessing, or user interface.")
st.write("Deployment:")
st.markdown("-Choose a deployment platform (e.g.,Streamlit cloud) to host the Streamlit application.")
st.markdown("-Deploy the application to make it accessible via a web browser.")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

