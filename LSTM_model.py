import numpy as np
import pandas as pd
import math
from keras.models import model_from_json
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

# Read CSV and data normalization is done in this function
def get_dataset(dataset_name,normalized = 0, file_name = None):
    print(file_name)
    print(dataset_name)
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

# Funcao que compila o modelo

def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#fase de teste

def evaluate_test(second,third,model,X_test,y_test,toMultiply):
    testScore = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0],math.sqrt(testScore[0])))
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(np.array(p))
    np.savetxt('prediction' + second + third + '.csv', np.transpose((y_test * toMultiply,predic * toMultiply)), delimiter=',',fmt="%s")
    with open('prediction' + second + third + '.csv', 'r+') as temp:
        content = temp.read()
        temp.seek(0,0)
        temp.write('Atual,Previsao\n')
        temp.write(content)

def LSTM_start(first,second,third):
    df , toMultiply = load_dataset(second)
    if(first == "Freguesia"):
        janela = 72
    else:
        janela = 24
    X_train, y_train, X_test, y_test = load_data(df, janela)
    #carregar o modelo
    model = load_model_json(second + third + "_model.json")
    #carregar os pesos
    load_weights_hdf5(model,second + third + "_model_weights.h5")
    #compila o modelo
    compile_model(model)
    #efetua a fase de teste
    evaluate_test(second,third,model,X_test,y_test,toMultiply)


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
    if(name == "S._Vitor"):
        name = "svitor"
    print(name)

    LSTM_start(freg_or_stree,name,long_or_short)

if __name__  == "__main__":
    main()
