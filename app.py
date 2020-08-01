# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: ASUS
"""
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (32,32)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('final_model.h5')

st.title("Waste-Management")

st.write("This is a simple image classification web app to predict the waste type")

img = Image.open('waste.png')
st.image(img, caption= 'Waste Segregator', use_column_width=True)

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a CardBoard!")
        st.write("### Hurray! It's a Biodegredable material")
        st.write("You are one step closer to saving the environment.")
        st.balloons()
    elif np.argmax(prediction) == 1:
        st.write("It is a Glass!")
        st.write("### Woops! It's a Non-Biodegredable material")
    elif np.argmax(prediction) == 2:
        st.write("It is a Metal!")
        st.write("### Woops! It's a Non-Biodegredable material")
    elif np.argmax(prediction) == 3:
        st.write("It is a Paper!")
        st.write("### Hurray! It's a Biodegredable material")
        st.write("You are one step closer to saving the environment.")
        st.balloons()
    elif np.argmax(prediction) == 4:
        st.write("It is a Plastic!")
        st.write("### Woops! It's a Non-Biodegredable material")
    else:
        st.write("It is a Trash!")
    
    st.text("Probability (0: CardBoard, 1: Glass, 2: Metal, 3: Paper, 4: Plastic, 5: Trash)")
    st.write(prediction)