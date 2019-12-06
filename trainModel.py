import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
size = 224
train_path = 'F:\programming\Imagenet/train'
valid_path='F:\programming\Imagenet/valid'
test_path='F:\programming\Imagenet/test'
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	train_path,
	target_size=(224,224),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	valid_path,
	target_size=(224,224),
	class_mode='categorical'
)

#train_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(size, size ), batch_size=10)
#valid_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(size, size), batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(size, size), batch_size=10)
mobile = keras.applications.mobilenet.MobileNet()
x=mobile.layers[-6].output
predictions = Dense(10, activation = 'softmax')(x)
model = keras.Model(inputs = mobile.input, outputs = predictions)
for layer in model.layers[:-40]:
    layer.trainable = False
model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=17, validation_data=validation_generator, validation_steps=3, epochs=150, verbose=2)

test_labels = test_batches.classes

# predictions = model.predict_generator(test_batches, steps=5, verbose=0)

# cm=confusion_matrix(test_labels, predictions.argmax(axis=1))
# test_batches.class_indices
# cm_plot_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
model.save('test92.h5')
