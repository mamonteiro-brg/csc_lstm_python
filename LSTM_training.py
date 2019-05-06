import numpy as np
import pandas as pd
import math,time
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import json

seed = 9
np.random.seed(seed)

def save_hiper_to_json(validation_split, epochs, batch_size,verbose,filename):
    dic = {'epochs':epochs,'batch_size': batch_size,'verbose': verbose, 'validation_split': validation_split}
    with open(filename,'w') as outfile:
        json.dump(dic,outfile)

def save_model_json(model,fich):
    model_json = model.to_json()
    with open(fich,"w") as json_file:
        json_file.write(model_json)

def save_weights_hdf5(model,fich):
    model.save_weights(fich)
    print("Save model to disk")

# Funcao para construcao do modelo

def build_model(janela):
    model = Sequential()
    model.add(LSTM(128,input_shape=(janela,3), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64,input_shape=(janela,3),  return_sequences=True))
    model.add(Dense(16,activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model

