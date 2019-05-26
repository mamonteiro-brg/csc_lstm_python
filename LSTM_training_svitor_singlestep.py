import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
import sys
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM
import json


from keras import backend as K


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

# Read CSV and data normalization is done in this function
def get_dataset(dataset_name,normalized = 0, file_name = None):
    print(file_name)
    print(dataset_name)

    # Estes são as novas features
    col_names = ["Mes","Dia","Ano","Hora","Minutos","Dia da semana","free_flow_distance","total_time","total_speed","diif_ranker","length_in_meters","delay_in_seconds","incidente","temperature","atmospheric_pressure","humidity","wind_speed","cloudiness","no_rain","chuva_fraca","chuva","chuva_trovoada","chuva_intensa"]

    stocks = pd.read_csv(file_name, header = 0, names = col_names)
    df = pd.DataFrame(stocks)

    #There are the fields that we have removed
    # mes -0
    # dia - 1
    # ano - 2
    # hora -3
    # minutos -4
    # dia da semana - 5

    df.drop(df.columns[[0,1,2,3,4,5]], axis = 1, inplace= True)


    #Encontrar os maximos de cada coluna
    maxFree_flow_distance = df['free_flow_distance'].max()
    maxTotal_time = df['total_time'].max()
    maxTotal_speed = df['total_speed'].max()
    maxDiff_ranker = df['diif_ranker'].max()
    maxLength_in_meters = df['length_in_meters'].max()
    maxDelay_in_seconds = df['delay_in_seconds'].max()
    maxIncidente = df['incidente'].max()
    maxTemperature = df['temperature'].max()
    maxAtmospheric_pressure = df['atmospheric_pressure'].max()
    maxHumidity = df['humidity'].max()
    maxWind_speed = df['wind_speed'].max()
    maxCloudiness = df['cloudiness'].max()
    maxNo_rain = df['no_rain'].max()
    maxChuva_fraca = df['chuva_fraca'].max()
    maxChuva = df['chuva'].max()
    maxChuva_trovoada = df['chuva_trovoada'].max()
    maxChuva_intensa = df['chuva_intensa'].max()


    df = df[["free_flow_distance","total_time","total_speed","diif_ranker","length_in_meters","delay_in_seconds","incidente","temperature","atmospheric_pressure","humidity","wind_speed","cloudiness","no_rain","chuva_fraca","chuva","chuva_trovoada","chuva_intensa"]]

    #df = df[['free_flow_distance','total_time','total_speed','length_in_meters','delay_in_seconds','incidente', 'temperature','atmospheric_pressure','humidity','wind_speed', 'cloudiness','no_rain','chuva_fraca', 'chuva', 'chuva_trovoada', 'chuva_intensa', 'diif_ranker']]
    #normalizar todas os atributos, e por na variavel to return qual o fator de normalizaçao da ultima coluna

    df['free_flow_distance'] = df['free_flow_distance']/maxFree_flow_distance
    df['total_time']=df['total_time']/maxTotal_time
    df['total_speed']=df['total_speed']/maxTotal_speed
    df['diif_ranker']=df['diif_ranker']/maxDiff_ranker
    df['length_in_meters']=df['length_in_meters']/maxLength_in_meters
    df['delay_in_seconds']=df['delay_in_seconds']/maxDelay_in_seconds
    df['incidente']=df['incidente']/maxIncidente
    df['temperature']=df['temperature']/maxTemperature
    df['atmospheric_pressure']=df['atmospheric_pressure']/maxAtmospheric_pressure
    df['humidity'] = df['humidity']/maxHumidity
    df['wind_speed']=df['wind_speed']/maxWind_speed
    df['cloudiness']=df['cloudiness']/maxCloudiness
    df['no_rain'] = df['no_rain']/maxNo_rain
    df['chuva_fraca']=df['chuva_fraca']/maxChuva_fraca
    df['chuva']=df['chuva']/maxChuva
    df['chuva_trovoada']= df['chuva_trovoada']/maxChuva_trovoada
    df['chuva_intensa']= df['chuva_intensa']/maxChuva_intensa

    toReturn = maxDiff_ranker

    print(df)
    return (df,toReturn)

def load_dataset(arg):
    dataset_name = arg
    print(dataset_name)
    return get_dataset(dataset_name, 0, dataset_name + '.csv')

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Funcao para construcao do modelo
def build_model(janela):

    model = Sequential()

    model.add(LSTM(128,input_shape=(janela,17), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(64,input_shape=(janela,17), return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(16,activation='sigmoid', kernel_initializer="uniform"))
    model.add(Dense(8, activation='sigmoid', kernel_initializer="uniform"))
    model.add(Dense(1, activation='linear',kernel_initializer="uniform"))

    model.compile(loss = root_mean_squared_error,optimizer='adam' ,metrics= ['accuracy'])
    return model


def load_data(df_dados,janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.as_matrix()
    tam_sequecia = janela
    res = []
    for i in range(len(mat_dados) - tam_sequecia + 1):
        res.append(mat_dados[i: i + tam_sequecia])
    res = np.array(res)

    print("Res")
    print(res)
    print("----------------------------")

    # 80% do dataset é usado para o treino do modelo
    # mas a quantos registos isso equivale? Como fazer para tratar um numero de registos correcto?
    # 28*24*numero freguesia
    # Dataset Total (13002, 17)

    qt_casos_treino = 10985
    print("Casos de Treino")
    print(qt_casos_treino)
    print("##################")
    train = res[:qt_casos_treino,:]
    x_train = train[:, :]
    y_train = train[:,-1][:,-1]

    x_test = res[qt_casos_treino:, :]
    y_test = res[qt_casos_treino:,-1][:,-1]
    x_train =  np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))
    return [x_train, y_train, x_test, y_test]

def print_series_prediction(y_test,predic):
    diff = []
    racio = []
    for i in range(len(y_test)):
        racio.append((y_test[i]/predic[i])-1)
        diff.append(abs(y_test[i]-predic[i]))
        #print('valor: %f ----> Previsao: %f  Diff: %f Racio: %f' % (y_test[i], predic[i],diff[i],racio[i]))

    print("prediction ")
    print(predic)
    print("-------------------------------------")

    plt.plot(y_test,color = 'blue', label = 'y_test')
    plt.plot(predic, color = 'red', label = 'prediction')
    plt.plot(diff, color = 'green', label = 'diff')
    plt.plot(racio, color = 'yellow', label = 'racio')
    plt.legend(loc = 'upper left')
    plt.show()

def print_model(model,fich):
    from keras.utils import plot_model
    plot_model(model,to_file= fich,show_shapes= True,show_layer_names= True)

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()

def evaluate_test(arg, model,X_test,y_test,toMultiply):
    testScore = model.evaluate(X_test,y_test,verbose=0)
    print('Test Socre: %.2f MSE (%.2f RMSE)' % (testScore[0],math.sqrt(testScore[0])))
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(np.asarray(p))
    print("prediction ")
    print(predic)
    print("-------------------------------------")

    print_series_prediction(y_test, predic)

def LSTM_start(arg):
    df, toMultiply = load_dataset(arg)
    janela = 72
    print("Dataset Total", df.shape)
    X_train, y_train, X_test, y_test = load_data(df,janela)
    print("X_train",X_train.shape)
    print("y_train",y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)


    # Train a new classifier using the best parameters found by the grid search
    model = build_model(janela)
    history = model.fit(X_train,y_train,batch_size=72,epochs=10,validation_split = 0.1,verbose= 1)


    save_model_json(model, arg + "_model.json")
    save_weights_hdf5(model, arg + "_model_weights.h5")
    save_hiper_to_json(0.1,1500,1,1,arg + "_hiperparametros.json")
    #print_model(model,"lstm_model.png")
    print_history_loss(history)
    trainScore = model.evaluate(X_train, y_train, verbose= 0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0],math.sqrt(trainScore[0])))
    evaluate_test(arg, model,X_test,y_test,toMultiply)

def main():
    arg = sys.argv[1]
    print(arg)
    LSTM_start(arg)
    #f.close()


if __name__ == "__main__":
    main()
