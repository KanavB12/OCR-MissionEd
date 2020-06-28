#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import keras
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Conv2DTranspose, Flatten
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
from keras.callbacks import ModelCheckpoint


# In[ ]:


import pickle

with open('/content/drive/My Drive/MissionEd/test_data.pickle', 'rb') as f:
  valid_images, valid_labels, valid_input_length, valid_label_length, valid_original_text = pickle.load(f)

with open('/content/drive/My Drive/MissionEd/train_data.pickle', 'rb') as f:
  train_images, train_labels, train_input_length, train_label_length, train_original_text = pickle.load(f)


# In[ ]:


train_images = np.asarray(train_images)
valid_images = np.asarray(valid_images)


# In[ ]:


lr_images = []
for i in range(0, len(train_images)):
  temp = cv2.resize(train_images[i].reshape(32, 128), (64, 16))
  lr_images.append(temp)


lr_test = []
for i in range(0, len(valid_images)):
  temp = cv2.resize(valid_images[i].reshape(32, 128), (64, 16))
  lr_test.append(temp)


# In[ ]:


hr_images = []
for i in range(0, len(train_images)):
  temp = train_images[i].reshape(32, 128)
  hr_images.append(temp)

hr_test = []
for i in range(0, len(valid_images)):
  temp = valid_images[i].reshape(32, 128)
  hr_test.append(temp)


# In[ ]:


lr_images = np.asarray(lr_images)
lr_test = np.asarray(lr_test)
hr_images = np.asarray(hr_images)
hr_test = np.asarray(hr_test)


# In[ ]:


X_train = lr_images.reshape(lr_images.shape[0], 16, 64, 1)
X_test = lr_test.reshape(lr_test.shape[0], 16, 64, 1)

y_train = hr_images.reshape(hr_images.shape[0], 4096)
y_test = hr_test.reshape(hr_test.shape[0], 4096)


# In[ ]:


inputs = Input(shape = (16, 64, 1))

# Conv 1 Layer
conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(inputs)

# Conv 2 Layer
conv_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', strides=1)(conv_1)

# ConvTranspose Layer
conv_3 = Conv2DTranspose(filters = 16, kernel_size=(3, 3), padding='same', strides = 2)(conv_2)

# Conv Layer
conv_4 = Conv2D(filters = 1, kernel_size=(3, 3), padding='same', strides = 1)(conv_3)

# Flatten
flat = Flatten()(conv_4)

res_model = Model(inputs, flat)
res_model.summary()


# In[ ]:


batch_size = 8
epochs = 20
e = str(epochs)
optimizer_name = 'adam'
loss = keras.losses.MeanSquaredError()


# In[ ]:


res_model.compile(loss=loss, optimizer = optimizer_name, metrics=['accuracy'])

filepath="{}o-{}e.hdf5".format(optimizer_name,
                                          str(epochs))

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]


# In[ ]:


history = res_model.fit(x=X_train,
                    y=y_train,
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=(X_test, y_test),
                    verbose=2,
                    callbacks=callbacks_list)


# In[ ]:


test_s = X_test[0:10]
test_s = np.asarray(test_s)

pred = res_model.predict(test_s)


# In[ ]:


pred = pred.reshape(10, 32, 128)


# In[ ]:


from sklearn.externals import joblib

joblib_file = "RES_Model.pkl"  
joblib.dump(res_model, joblib_file)


# In[ ]:





# In[ ]:




