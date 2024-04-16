#!/usr/bin/env python
# coding: utf-8

# Importing all the required Libraries

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense,Conv2D,GlobalAvgPool2D,Input
from tensorflow.keras import callbacks,optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


# In[2]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[3]:


get_ipython().system('kaggle datasets download -d alessiocorrado99/animals10')


# In[4]:


import zipfile
zip_ref = zipfile.ZipFile('/content/animals10.zip', 'r')
zip_ref.extractall('/content/Animal')
zip_ref.close()


# In[5]:


get_ipython().run_line_magic('cd', '/content/Animal')


# In[6]:


from translate import translate


# In[7]:


translate


# In[8]:


os.listdir("/content/Animal/raw-img")


# In[9]:


#manually renaming specific directory
for i in os.listdir("/content/Animal/raw-img"):
  try:
    os.rename("/content/Animal/raw-img/"+i,"/content/Animal/raw-img/"+translate[i])
  except Exception as e:
    print(e)
os.rename("/content/Animal/raw-img/ragno","/content/Animal/raw-img/spider")
#os.rename("/content/Animal/raw-img/scoiattolo","/content/Animal/raw-img/squirrel")


# In[10]:


ls /content/Animal/raw-img


# In[11]:


for i in os.listdir("/content/Animal/raw-img"):
  print(i,len(os.listdir("/content/Animal/raw-img/"+i)))


# In[12]:


try:
  os.mkdir("train")
  os.mkdir("test")
except:
  pass
for i in os.listdir("/content/Animal/raw-img"):
  try:
    os.mkdir("train/"+i)
    os.mkdir("test/"+i)
  except:
    pass
  for j in os.listdir("/content/Animal/raw-img/"+i)[:1000]:
    os.rename("/content/Animal/raw-img/"+i+"/"+j,"train/"+i+"/"+j)
  for j in os.listdir("/content/Animal/raw-img/"+i)[:400]:
    os.rename("/content/Animal/raw-img/"+i+"/"+j,"test/"+i+"/"+j)


# Set path to Dataset

# In[13]:


train_dir = '/content/Animal/train'
validation_dir = '/content/Animal/test'


# In[14]:


len(os.listdir("/content/Animal/raw-img"))


# Setting up th Paramaters

# In[15]:


num_classes = 10
image_size = (224, 224)
batch_size = 32
learning_rate = 0.001
epochs = 5


# Preprocessing and augument the training data

# In[16]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# Preprocess the validation data

# In[17]:


valid_datagen = ImageDataGenerator(rescale=1./255)


# Load the inceptionv3 model

# In[18]:


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))


# Adding the top layer for the model

# In[19]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(500, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


# Combine base model with top layer

# In[20]:


model = Model(inputs=base_model.input, outputs=predictions)


# Freeze the layers in base model

# In[21]:


for layer in base_model.layers:
  layer.trainable = False


# Compile the model

# In[22]:


model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


# Generate the training and validation data from directories

# In[23]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# In[24]:


train_generator.class_indices


# In[25]:


base_model.trainable = False


# Train the model

# In[26]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // batch_size
)


# In[27]:


# Model Evaluation
evaluation = model.evaluate(valid_generator, steps=valid_generator.n // batch_size)

# Displaying the Evaluation Metrics
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

# Plotting Training and Validation Curves
plt.figure(figsize=(12, 4))

# Plotting Training Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[28]:


model.save("/content/Animal_Classification.h5")


# Testing the model

# In[29]:


import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('/content/Animal_Classification.h5')


class_labels = ['butterfly' , 'cat' , 'chicken' , 'cow' , 'dog' , 'elephant' , 'horse' , 'sheep' , 'spider' , 'squirrel']


def preprocess_image(image_path):
    img = PIL.Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_animal(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence


from google.colab import files
uploaded = files.upload()


uploaded_file_path = list(uploaded.keys())[0]


prediction, confidence = predict_animal(uploaded_file_path)


img = PIL.Image.open(uploaded_file_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Prediction: {prediction}\nConfidence: {confidence:.2%}')
plt.show()


# In[30]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import cv2\nimport numpy as np\nimport streamlit as st\nimport tensorflow as tf\nfrom tensorflow.keras.preprocessing import image\nfrom tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input\n\nst.title("Animal Image Detection Application")\nst.header("Image Classifier ")\n\nmodel = tf.keras.models.load_model("/content/Animal_Classification.h5")\n### load file\nuploaded_file = st.file_uploader("Upload an Image for Image classification", type=["jpg","jpeg","png"])\n\nmap_dict = { 0: \'butterfly\',\n             1: \'cat\',\n             2: \'chicken\',\n             3: \'cow\',\n             4: \'dog\' ,\n             5: \'elephant\',\n             6: \'horse\',\n             7: \'sheep\',\n             8: \'spider\',\n             9: \'squirrel\'}\n\n\nif uploaded_file is not None:\n    # Convert the file to an opencv image.\n    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)\n    opencv_image = cv2.imdecode(file_bytes, 1)\n    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)\n    resized = cv2.resize(opencv_image,(224,224))\n    # Now do something with the image! For example, let\'s display it:\n    st.image(opencv_image, channels="RGB")\n\n    resized = mobilenet_v2_preprocess_input(resized)\n    img_reshape = resized[np.newaxis,...]\n\n    Genrate_pred = st.button("Generate Prediction")\n    if Genrate_pred:\n        prediction = model.predict(img_reshape).argmax()\n        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))\n')


# In[31]:


get_ipython().system('pip install streamlit --quiet')


# In[ ]:


get_ipython().system('streamlit run app.py & npx localtunnel --port 8501')

