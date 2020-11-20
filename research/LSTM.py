# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import losses
import math
import numpy as np
import scipy.io as sio


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def get_data(path):
    data_struct = sio.loadmat(path)
    data = data_struct['data']
    ##you need to sample defects and some background pxiels to make them balance to some degree
    #and return train and label with numpy format
    return train, label


train, label = get_data(path='')
label = tf.keras.utils.to_categorical(label) #make label from one column to two cloums;
# first is zero(in this label,zero is background), second is one
training_batch_size = train.shape[0]
training_timestep = train.shape[1]
training_feature_number = 1
training_class_number = 2

HIDDEN_LAYER_SIZE = 16

y_hu = np.reshape(label, [training_batch_size, training_class_number])
x_hu = np.reshape(train, [training_batch_size, training_timestep, training_feature_number])  #train and train_


sgd = SGD(lr=0.0,momentum=0.9,decay=0.0,nesterov=False)
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
BATCH_SIZE = 400  #200,400

model = Sequential()
model.add(LSTM(HIDDEN_LAYER_SIZE,return_sequences=True,input_shape=(training_timestep,training_feature_number)))
model.add(LSTM(HIDDEN_LAYER_SIZE,return_sequences=True))
model.add(LSTM(HIDDEN_LAYER_SIZE))
model.add(Dropout(0.5))
model.add(Dense(training_class_number,activation='softmax'))
print(model.summary())
model.compile(loss = losses.mean_squared_error, optimizer = sgd,metrics=["accuracy"])

NUM_EPOCHS = 20   #20
model.fit(x_hu, y_hu, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=callbacks_list)
model.save('./F17/150/model.h5')
# json_s = model.to_json()
# open('./model/RC200train.json','w').write(json_s)
# model.save_weights('./model/RC200train.h5')