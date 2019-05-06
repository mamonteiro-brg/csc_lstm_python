import numpy as np
import pandas as pd
import math,time
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import json
import sys


seed = 9
np.random.seed(seed)


def load_hiper(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def load_model_json(fich):
    json_file = open(fich,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def load_weights_hdf5(model,fich):
    model.load_weights(fich)
    print("Loaded model from disk")

# Funcao que compila o modelo

def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
