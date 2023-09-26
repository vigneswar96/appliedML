#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np
from PIL import Image
import os

model2_path = './model2_squirrelornot.h5'
model2 = tf.keras.models.load_model(model2_path)
model3_path = './model3_birdfeederornot.h5'
model3 = tf.keras.models.load_model(model3_path)

directory_path = 'Downloads/test_images/'
file_paths = []  # List to store user-provided file paths

# Prompt the user to enter a list of file paths
file_paths_input = input("Enter a comma-separated list of file paths: ")
file_paths = file_paths_input.split(",")

selected_images = []
count = 0

for file_path in file_paths:
    file_path = file_path.strip()  # Remove any leading/trailing spaces
    if os.path.isfile(file_path) and (file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.JPEG')):
        image = Image.open(file_path).resize((299, 299))
        image_array = np.array(image) / 255.0
        image_tensor = np.expand_dims(image_array, axis=0)

        prediction_model2 = model2.predict(image_tensor)
        predicted_label_model2 = np.argmax(prediction_model2, axis=1)[0]

        prediction_model3 = model3.predict(image_tensor)
        predicted_label_model3 = np.argmax(prediction_model3, axis=1)[0]

        if predicted_label_model2 == 0 and predicted_label_model3 == 0:
            selected_images.append(file_path)
            count += 1
            if count >= 5:
                break
            print("Image:", file_path)
            print("Model 2 predicted label:", predicted_label_model2)
            print("Model 3 predicted label:", predicted_label_model3)
            print()
    else:
        print("Invalid file path:", file_path)

