import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import load_model 
import os 
import h5py 

st.header("marbling Sorter")

def main():
    file_uploaded=st.file_uploader("choose the file", type= ['jpg','png','jpeg'])
    if file_uploaded is not None:  
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result= predict_class(image)
        st.write(result)
        st.write(figure)
        
def predict_class(image):
    classifier_model= tf.keras.models.load_model(r'/content/drive/MyDrive/code/model_save/mymodel.hdf5')
    shape=((128,128,3))
    model= tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
    test_image=image.resize((128,128))
    test_image=preprocessing.image.img_to_array(test_image)
    test_image=test_image/255.0
    test_image= np.expand_dims(test_image, axis=0)
    class_names=['G1','G2','G3','G4','G5','G6','G7','G8']
    predictions= model.predict(test_image)
    scores=tf.nn.softmax(predictions[0])
    scores=scores.numpy()
    image_class=class_names[np.argmax(scores)]
    results= "The photo you have uploaded is:{}".format(image_class)
    return results 
if __name__=="__main__":
    main()

