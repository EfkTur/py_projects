import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

annotations_dir = './annotation/Annotation/'
images_dir = './images/Images'

def show_images_classes(path, classes, num_sample):
    """
    Raison d'être:
        This function is used to display the first 
        n images of a directory passed as an argument. 
        It is adapted to subdirectories. 
        
        The matplotlib.image library must be loaded 
        with the alias mpimg. 

    Args
    ----------------------------------------
        path : string
            Link of root directory
        classes : string 
            Name of the subdirectory
        num_smaple : integer
            Number of picture to show
    ----------------------------------------

    Returns:
        None. Shows a plot.
    """
    fig = plt.figure(figsize=(20,20))
    fig.patch.set_facecolor('#343434')
    plt.suptitle("{}".format(classes.split("-")[1]), y=.83, color="white", fontsize=22)
    images = os.listdir(path + "/" + classes)[:num_sample]
    for i in range(num_sample):
        img = mpimg.imread(path+"/"+classes+"/"+images[i])
        plt.subplot(int(num_sample/5+1), 5, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def plot_histogram(init_img, convert_img):
    """
    Raison d'être:
        Function allowing to display the initial
        and converted images according to a certain
        colorimetric format as well as the histogram
        of the latter. 

    Args:
        -------------------------------------------
        init_img : list
            init_img[0] = Title of the init image
            init_img[1] = Init openCV image
        convert_img : list
            convert_img[0] = Title of the converted
            convert_img[1] = converted openCV image
        -------------------------------------------
    Returns:
        None
    """
    
    
    hist, bins = np.histogram(
                    convert_img[1].flatten(),
                    256, [0,256])
    # Cumulative Distribution Function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Plot histogram
    fig = plt.figure(figsize=(25,6))
    plt.subplot(1, 3, 1)
    plt.imshow(init_img[1])
    plt.title("{} Image".format(init_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 2)
    plt.imshow(convert_img[1])
    plt.title("{} Image".format(convert_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 3)
    plt.plot(cdf_normalized, 
             color='r', alpha=.7,
             linestyle='--')
    plt.hist(convert_img[1].flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title("Histogram of convert image", color="#343434")
    plt.suptitle("Histogram and cumulative "\
                 "distribution for test image",
              color="black", fontsize=22, y=.98)
    plt.show()

def preprocessing_cnn(directories, img_width, img_height):
    """
    Raison d'être:
    
        Preprocessing of images in order to integrate them 
        into a convolutional neural network. Equalization, 
        Denoising and transformation of the image into Array. 
        Simultaneous creation of labels (y). 

    Args
        ---------------------------------------------------
        directoriesList : list
            List of files to be processed.
        img_width : integer
            width of the image to be reached for resizing
        img_height : integer
            height of the image to be reached for resizing
        ---------------------------------------------------
    Returns:
    img_list: List of images after preprocessing 
    labels: the annotations corresponding to these images
    
    """
    images_dir = './images/Images'
    img_list=[]
    labels=[]
    for index, breed in enumerate(directories):
        for image_name in os.listdir(images_dir+"/"+breed):
            # Read image
            img = cv.imread(images_dir+"/"+breed+"/"+image_name)
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            # Resize image
            dim = (img_width, img_height)
            img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
            # Equalization
            img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
            img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
            # Apply non-local means filter on test img
            dst_img = cv.fastNlMeansDenoisingColored(
                src=img_equ,
                dst=None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21)
            
            # Convert modified img to array
            img_array = image.img_to_array(dst_img)
            
            # Append lists of labels and images
            img_list.append(np.array(img_array))
            labels.append(breed.split("-")[1])
    
    return img_list, labels


def plot_history_scores(dict_history, first_score, second_score):
    """
    Raison d'être:
        Function to plot the results of our various neural networks
    
    Args:
        Dict_history: A dictonary containing the various values of performances over the epochs
        first_score: First metric to be used to assess performances
        second_score: Second metric to be used to assess performances
    
    Returns:
        None
    """
    
    
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(25,10))
        # summarize history for accuracy
        plt.subplot(1, 2, 1)
        plt.plot(dict_history.history[first_score], color="g")
        plt.plot(dict_history.history['val_' + first_score],
                 linestyle='--', color="orange")
        plt.title('CNN model ' + first_score, fontsize=18)
        plt.ylabel(first_score)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(dict_history.history[second_score], color="g")
        plt.plot(dict_history.history['val_' + second_score],
                 linestyle='--', color="orange")
        plt.title('CNN model ' + second_score, fontsize=18)
        plt.ylabel(second_score)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def recall_me(y_true, y_pred):
    """
    Raison d'être:
        Calculates the recall metric
    
    Args:
        y_true: Our actual labels
        y_pred: Our predicted labels
    
    Returns:
        recall: A statistic
    """
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_me(y_true, y_pred):
    """
    Raison d'être:
        Calculates the precision metric
    
    Args:
        y_true: Our actual labels
        y_pred: Our predicted labels
    
    Returns:
        precision: A statistic
    """
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    """
    Raison d'être:
        Calculates the f1_score metric
    
    Args:
        y_true: Our actual labels
        y_pred: Our predicted labels
    
    Returns:
        f1_score: A statistic
    """
    
    precision = precision_me(y_true, y_pred)
    recall = recall_me(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))