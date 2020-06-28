# -*- coding: utf-8 -*-
"""Word_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mBwui73g7a7NavkwkZ_RGQS8ic_MugYZ
"""

import tensorflow as tf
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

from google.colab import drive
drive.mount('/content/drive')

# Loading train and test pickled data

import pickle

with open('/content/drive/My Drive/MissionEd/test_data.pickle', 'rb') as f:
  valid_images, valid_labels, valid_input_length, valid_label_length, valid_original_text = pickle.load(f)

with open('/content/drive/My Drive/MissionEd/train_data.pickle', 'rb') as f:
  train_images, train_labels, train_input_length, train_label_length, train_original_text = pickle.load(f)

len(train_images)

"""### Import Libraries"""

import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Conv2DTranspose, Flatten
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""# Preprocess Data"""

max_label_len = 0

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# string.ascii_letters + string.digits (Chars & Digits)
# or 
# "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

print(char_list, len(char_list))

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))
        
    return dig_lst

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape
    
    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w), interpolation = cv2.INTER_AREA)
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img

"""## Generate Train and Validate set"""

# setting max length to 16
max_label_len = 16

# padded_label = pad_sequences(labels, 
#                              maxlen=max_label_len, 
#                              padding='post',
#                              value=len(char_list))

train_padded_label = pad_sequences(train_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

valid_padded_label = pad_sequences(valid_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

train_padded_label.shape, valid_padded_label.shape

"""## Convert to numpy array"""

train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

valid_images = np.asarray(valid_images)
valid_input_length = np.asarray(valid_input_length)
valid_label_length = np.asarray(valid_label_length)

"""## Build Model"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from keras.backend import squeeze

inputs = Input(shape = (32, 128, 1))

# Conv Layer 1
conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(inputs)
pool_1 = MaxPool2D(pool_size = (2, 2))(conv_1)

# Conv Layer 2
conv_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(pool_1)
pool_2 = MaxPool2D(pool_size = (4, 2))(conv_2)

# Conv Layer 3
conv_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(pool_2)
norm = BatchNormalization()(conv_3)

# Conv Layer 4
conv_4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(norm)
pool_4 = MaxPool2D(pool_size = (4, 1))(conv_4)

# Squeeze
squeezed = Lambda(lambda x: K.squeeze(x, 1))(pool_4)

# bidirectional LSTM layers with units=128
blstm = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)

# Ouptut Layer
outputs = Dense(len(char_list) + 1, activation = 'softmax')(blstm)

# Model to be used at test time
act_model = Model(inputs, outputs)

act_model.summary()

the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)

batch_size = 8
epochs = 20
e = str(epochs)
optimizer_name = 'sgd'

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer_name, metrics=['accuracy'])

filepath="{}o-{}e-{}t-{}v.hdf5".format(optimizer_name,
                                          str(epochs),
                                          str(train_images.shape[0]),
                                          str(valid_images.shape[0]))

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length],
                    y=np.zeros(len(train_images)),
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=([valid_images, valid_padded_label, valid_input_length, valid_label_length], [np.zeros(len(valid_images))]),
                    verbose=2,
                    callbacks=callbacks_list)

"""## Test Accuracy"""

act_model.load_weights('/content/sgdo-20e-86818t-9636v.hdf5')

# predict outputs on validation images
prediction = act_model.predict(valid_images)
 
# use CTC decoder
decoded = K.ctc_decode(prediction, 
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]
out = K.get_value(decoded)

"""## Saving the model"""

from sklearn.externals import joblib

joblib_file = "OCR_Model.pkl"  
joblib.dump(act_model, joblib_file)

pip install python-Levenshtein

import Levenshtein as lv

total_jaro = 0
total_rati = 0
# see the results
for i, x in enumerate(out):
    letters=''
    for p in x:
        if int(p) != -1:
            letters+=char_list[int(p)]
    total_jaro+=lv.jaro(letters, valid_original_text[i])
    total_rati+=lv.ratio(letters, valid_original_text[i])

print('jaro :', total_jaro/len(out))
print('ratio:', total_rati/len(out))

"""## Train Accuracy"""

prediction = act_model.predict(train_images[540:560])
 
# use CTC decoder
decoded = K.ctc_decode(prediction,   
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]

out = K.get_value(decoded)

# see the results
for i, x in enumerate(out):
    print("original_text =  ", train_original_text[540+i])
    print("predicted text = ", end = '')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end = '')
    plt.imshow(train_images[540+i].reshape(32,128), cmap=plt.cm.gray)
    plt.show()
    print('\n')

"""## Testing on custom Input"""

import os

list = []

for filename in os.listdir():
  if filename.endswith('.jpeg'):
    list.append(filename)

list

## FOR TESTING ONE IMAGE

# Reading the Image
image = cv2.imread('/content/Bhavana1.jpeg')

# Converting to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Converting to a Binary Image
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Preprocessing the image
IMG = process_image(thresh)

Test_set = [IMG]
Test_set = np.asarray(Test_set)

# FOR TESTING MULTIPLE IMAGES
Test_set = []

for i in range(0, len(list)):
  img_path = '/content/{}'.format(list[i])
  
  # Reading the Image
  image = cv2.imread(img_path)

  # Converting to Grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Converting to a Binary Image
  ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

  # Preprocessing the image
  IMG = process_image(thresh)

  Test_set.append(IMG)
  
Test_set = np.asarray(Test_set)

Test_set.shape

# Predicting
Test_pred = act_model.predict(Test_set)

# use CTC decoder
test_decoded = K.ctc_decode(Test_pred,   
                       input_length=np.ones(Test_pred.shape[0]) * Test_pred.shape[1],
                       greedy=True)[0][0]

Test_out = K.get_value(test_decoded)

# see the results
for i, x in enumerate(Test_out):
  print("original text = ", list[i])
  print("predicted text = ", end = '')
  for p in x:
      if int(p) != -1:
          print(char_list[int(p)], end = '')
  plt.imshow(Test_set[i].reshape(32,128), cmap = 'gray')
  plt.show()
  print('\n')



