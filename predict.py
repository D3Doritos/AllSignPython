import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
#model = tf.keras.models.load_model('/home/ss/Documents/AllSignPython2/trained_model/withbgmodelv12.h5')
model = tf.keras.models.load_model('/home/ss/Documents/mnist/model/test92.h5')
size = 224
lpath='/home/ss/Documents/mnist/allnumbers'
a_dir = os.listdir(lpath)
  # predicting images
for fn in a_dir:
    path=fn
    
    img = tf.keras.preprocessing.image.load_img('/home/ss/Documents/mnist/allnumbers/'+path, target_size=(size, size))
    
    y = tf.keras.preprocessing.image.img_to_array(img)
    #img = cv2.resize(img, (50, 50))
    #x = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    x = y.reshape(-1, size, size, 3)
    
    
    #x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
        
        
    preds = model.predict(images, batch_size=10)
    preds *= 100
    most_likely_class_index = int(np.argmax(preds))
    
        #print(preds)
    print(fn)
    print(most_likely_class_index)
    print(preds)