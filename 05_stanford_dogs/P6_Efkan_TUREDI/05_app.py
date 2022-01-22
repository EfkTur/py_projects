#If runnung on a cloud AWS terminal make sure to install tensorflow, opencv (headless version), sklearn, matplotlib

import os
import tensorflow as tf
import cv2 as cv
import os
from tensorflow.keras.models import load_model
from stanford_dogs_utils import f1
from tensorflow import keras
import pickle


# Load model and encoder
model = load_model('./my_tuned_xcept_model.h5',custom_objects = {"f1": f1})
with open("./encoder.pickle", "rb") as f: 
    encoder = pickle.load(f)
    
num_breeds = len(encoder.classes_)

# Define the full prediction function
def breed_prediction(inp):
    """
    Raison d'être:
    Guess the class for an input dog image

    Args:
    inp: the image to be classified

    Returns:
    max_class: Predicted class of the image
    
    """
    #Read the input
    inp = cv.imread(inp)
    
    # Convert to RGB
    img = cv.cvtColor(inp,cv.COLOR_BGR2RGB)
    
    # Resize image
    dim = (299, 299)
    img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
    
    # Equalization
    img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    
    # Apply non-local means filter on img
    dst_img = cv.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21)

    # Convert modified img to array
    img_array = keras.preprocessing.image.img_to_array(dst_img)
    
    # Apply preprocess Xception
    img_array = img_array.reshape((-1, 299, 299, 3))
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    
    # Predictions
    prediction = model.predict(img_array).flatten()
    
    #print and return prediction
    results = {encoder.classes_[i]: float(prediction[i]) for i in range(num_breeds)}
    max_class = max(results,key=results.get)
    
    print('The dog you submitted is predicted as a: ',max_class)
    return max_class

breed_prediction('input.jpeg')
