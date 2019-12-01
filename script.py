import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import cv2
import argparse
import numpy as np
from variables import *
from keras.models import load_model
from flask_cors import CORS
import keras
import base64
def get_model():
    global model
    model = load_model(MODEL_PATH)
    model._make_predict_function()
    
    print("Model Loaded")

def preprocess_image(imgIn):
            
        try:
            img = readb64(imgIn)
        
            # Change color sapce to gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Reshape array to l * w * channels
            img_array = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
            # Normalize th array
            img_array = img_array / 225.0

            # Expand Dimension of the array as our model expects a 4D array
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print("Unexpected error", e)


def which_letter(imgIn):
    """
    :param IMG_PATH: path of the image
    :return: confident_percent, predicted label using the model or None if exception occurs
    eg:
        print(which_letter("sample.jpeg"))
    """
    get_model()
    try:
        img_array = preprocess_image(imgIn)
        preds = model.predict(img_array)
        preds *= 100
        most_likely_class_index = int(np.argmax(preds))
        print(preds)
        return preds.max(), LABELS[most_likely_class_index]

    except Exception as e:
        print(e)
        return None
def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img
#creating instance of the class
app=Flask(__name__)
CORS(app)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        # print('post', request.get_json()['imgSrc'])
        keras.backend.clear_session() 
        imgIn = request.get_json()['imgSrc']
        resOut = which_letter(imgIn)
        #print(which_letter())
        # conf, label = which_letter()
        # print("The predicted letter is {} with {}%  confidence".format(label, conf))
#to tell flask what url shoud trigger the function index()


        return str(resOut)