# Import libraries
from google.colab import drive
import os
import pickle
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle
import math 
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import urllib.request


def concat_dataset(path, img_size = 224):
  '''
  Resize imgs and concatanete imgs from all the categories into one dataset.

  Parameters
  ----------
  path : string
      Path to dataset folder.
  img_size : int
      Size of image after resizing. Default is 224.

  Returns
  -------
  X : numpy array 
      Contains all the data from different categories.
  y : numpy array
      Containts labels for the data.
  '''

  X = [] 
  y = []

  if os.path.isdir(path):  # check if directory exist and get all filenames
    for foldername in os.listdir(path):

      dirname  = path + foldername
      print(dirname)
      
      if os.path.isdir(dirname):
        for filename in os.listdir(dirname):

            filedir = dirname + "/" + filename
            print(filedir)

            img = cv2.imread(filedir)
            img_resized = img_to_array(array_to_img(img, scale = False).resize((img_size, img_size))) # default is linear interpolation
            img_color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # convert image to RGB color space
            
            X.append(img_color)
            y.append(int(foldername)) # label is contained in folder name

  return np.asarray(X), np.asarray(y)


def load_data(path, filename):
  '''
  Loads data from a pickle file.

  Parameters
  ----------
  data: numpy array
  path : string
      Path to folder containing the data.
  filename : string
      Name the file to be saved under.
      
  Returns
  -------
  data : numpy array
  '''

  with open(path + filename, 'rb') as output:
    data = pickle.load(output)

  return data


def save_data(data, path, filename):
  '''
  Saves data to a pickle file.

  Parameters
  ----------
  data : numpy array
      Data to be saved.
  path : string
      Path to folder.
  filename : string
      Name the file to be saved under.

  Returns
  -------
  None

  '''

  with open(path + filename, 'wb') as output:
    pickle.dump(data, output)
    

def get_prediction(image, model, class_names):
  '''
  Predicts image class.
  
  Parameters
  ----------
  image : image object
  model : deep learning model
      
  Returns
  -------
  class_names : list
    Predicted class.
  '''

  image = np.expand_dims(image, axis=0)
  
  prediction = model.predict(image)

  predicted_class = np.argmax(prediction)
  
  return class_names[predicted_class]


def url_to_image(url):
  '''
  Reads image from url and performs preprocessing.
   
  Parameters
  ----------
  url: string
      Url to image.
      
  Returns
  -------
  image_resized: image
    Preprocessed image.
  image : image
    Original image.
  '''

  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")

  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image_resized  = cv2.resize(image, (224,224))
  image_resized = image_resized/255.0

  return image_resized, image 


def predict_url(url, model, class_names):
  '''
  Shows image and predicted class.
  
  Parameters
  ----------
  url: string
      Url to image.
  model : deep learning model
  class_names : list
  

  Returns
  -------
  None

  '''
  image_resized, image = url_to_image(url)
  predicted_class = get_prediction(image_resized, model, class_names)
  plt.imshow(image_resized)

  plt.title("Predicted label : " + predicted_class)
