import numpy as np
import pandas as pd
import math,time
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import json
import os.path
import sys
import collections
import csv

seed = 9
np.random.seed(seed)

def save_hiper_to_json(validation_split, epochs, batch_size,verbose,filename):
    dic = {'epochs':epochs,'batch_size': batch_size,'verbose': verbose, 'validation_split': validation_split}
    with open(filename,'w') as outfile:
        json.dump(dic,outfile)

def load_hiper(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def save_model_json(model,fich):
    model_json = model.to_json()
    with open(fich,"w") as json_file:
        json_file.write(model_json)

def load_model_json(fich):
    json_file = open(fich,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def save_weights_hdf5(model,fich):
    model.save_weights(fich)
    print("Save model to disk")

def load_weights_hdf5(model,fich):
    model.load_weights(fich)
    print("Loaded model from disk")

# Funcao que compila o modelo

def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

#fase de teste

def evaluate_test(arg,model,X_test,y_test):
    testScore = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0],math.sqrt(testScore[0])))
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(np.array(p))


def LSTM_start(first,second,third):
    #carregar o dataset
    #df = load
    #carregar o modelo
    model = load_model_json(first + second + third + "_model.json")
    #carregar os pesos
    load_weights_hdf5(model,first + second + third + "_model.h5")
    #compila o modelo
    compile_model(model)
    #efetua a fase de teste
    #evaluate ( ... parametros ...)


def main():
    total = sys.argv
    print(total)
    #primeiro argumento saber se e a longo prazo ou nao
    long_or_short = sys.argv[1]
    #segundo argumento se e freguesia ou rua
    freg_or_stree = sys.argv[2]
    #terceiro argumento o nome da freguesia ou da rua
    name = sys.argv[3]
    print(long_or_short)
    print(freg_or_stree)
    print(name)

if __name__  == "__main__":
    main()
