import tensorflow as tf
import streamlit as st
import tensorflow.keras.preprocessing.image 
from PIL import Image, ImageOps
import numpy as np
import cv2
import json

#load model 

model = tf.keras.models.load_model('model1.h5')

#headers 
st.write("""
         # A Good Ol Image Classifier
         """
         )
st.write("This is a simple image classification web app to classify your image into categories")
file = st.file_uploader("Please upload an image file", type=["jpg"])

def import_and_predict(image_data, model):
    
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction


# def import_and_predict(image_data, model):
#     	#is this processing right 
#     	#preprocess image 
#         image = tensorflow.keras.preprocessing.image.img_to_array(image_data)
#         img_reshape = image / 255.0
#         prediction = model.predict(np.expand_dims(img_reshape, 0))
        
#         return prediction
if file is None:
    st.text("Please upload an image file")
else:
	#am i resizing this properly 
	#open and resize the image 
    image = Image.open(file)
   
    st.image(image, use_column_width=True)

    #predict 
    prediction = import_and_predict(image, model)
    st.write(prediction)
    prediction_probs = int(np.argmax(prediction.reshape(-1)))

    #find it in the json 
    f = open("mapping.json")
    class_d = json.load(f)
    key_list = list(class_d.keys())
    val_list = list(class_d.values())
    position  = val_list.index(prediction_probs)
    classid = key_list[position]

    #now find the class in the words file 
    searchfile = open("words.txt", "r")
    for line in searchfile: 
    	if classid in line: 
    		things = line 
    		print(things)
    thing = things[10:]
    searchfile.close()

   
    st.text("class identifier:")
    st.write(classid)
    st.text("Calculating...")
    st.text("This is likely a")
    st.write(thing)
  