#importing the necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import streamlit as st


model=load_model(r"C:\Users\HP\OneDrive\Desktop\Computer Vision\Animal Classification\dog_cat_snake_classifier.h5")


st.header("This is a Dog ,Cat ,Snake classifier")
st.write("Upload a image of dog or cat or snake")


upload_image=st.file_uploader("choose image ...",type=["jpg","jpeg","png"])


if upload_image is not None:
    image=Image.open(upload_image)
    st.image(image,caption="uploaded_image",width=100)
    target_size=(256,256)
    
    if image.size != target_size:
        image=image.resize(target_size)
        
    image_array=img_to_array(image)
    image_array=np.expand_dims(image_array,axis=0)



predict=st.button("Predict")



if predict:
    result=model.predict(image_array)
    pred=np.argmax(result[0])
    if pred==0:
        st.write("Its Cat")
    elif pred==1:
        st.write("Its Dog")
    else:
        st.write("Its Snake")
        
#Just ignor this below  code .....I just want to check the prediction for a particular image
#x=cv2.imread(r"C:\Users\HP\OneDrive\Desktop\Computer Vision\Animal Classification\test\snakes\2_0756.jpg")
##x_test=x.reshape(1,256,256,3)
#result_array=model.predict(x_test)
#pred=np.argmax(result_array[0])
#print("image id of :", result_array[0])
#print(pred)
