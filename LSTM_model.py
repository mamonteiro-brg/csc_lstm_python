import numpy as np
import pandas as pd
import math
from keras.models import model_from_json
import json
import sys
from sklearn.preprocessing import MinMaxScaler


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


    col_names = ["Mes","Dia","Ano","Hora","Minutos","Dia da semana","free_flow_distance","total_time","total_speed","length_in_meters","delay_in_seconds","incidente","temperature","atmospheric_pressure","humidity","wind_speed","cloudiness","no_rain","chuva_fraca","chuva","chuva_trovoada","chuva_intensa","diif_ranker"]

    stocks = pd.read_csv(file_name, header = 0, names = col_names)
    df = pd.DataFrame(stocks)

    #There are the fields that we have removed
    # mes -0
    # dia - 1
    # ano - 2
    # hora -3
    # minutos -4
    # dia da semana - 5

    #df.drop(df.columns[[0,1,2,3,4,5]], axis = 1, inplace= True)

    print("DF")
    print(df)
    print("-------------")



    values = df.values.reshape((len(df.values),23))

    print(values.shape)

    scaler_diff_ranker = MinMaxScaler(feature_range=(-1,1))

    diff_ranker_data = np.array(df['diif_ranker'].values).reshape(len(df['diif_ranker'].values),1)
    sdr = scaler_diff_ranker.fit_transform(diff_ranker_data)
    ranker_shape = diff_ranker_data.shape
    print(ranker_shape)

    scaler = MinMaxScaler(feature_range=(-1,1))
    normal = scaler.fit_transform(values)
    print(normal)


    return (normal,scaler, scaler_diff_ranker)


def load_dataset(arg):
    dataset_name = arg
    print(dataset_name)
    return get_dataset(dataset_name, 0, dataset_name + '.csv')

def load_data(df_dados,janela):


    # mas a quantos registos isso equivale? Como fazer para tratar um numero de registos correcto?
    # 28*24*numero freguesia
    # Dataset Total (13002, 17)

    tam_sequecia = janela
    res = []
    for i in range(len(df_dados) - tam_sequecia + 1):
        res.append(df_dados[i: i + tam_sequecia])
    res = np.array(res)

    qt_casos_treino = 3661
    print("Casos de Treino")
    print(qt_casos_treino)
    print("##################")
    train = res[:qt_casos_treino, :]
    x_train = train[:, :]
    y_train = train[:, -1][:, -1]

    x_test = res[qt_casos_treino:, :]
    y_test = res[qt_casos_treino:, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 23))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 23))
    return [x_train, y_train, x_test, y_test]

# Funcao que compila o modelo

def compile_model(model):
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


#fase de teste

def evaluate_test(second,third,model,X_test,y_test,toMultiply):
    testScore = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0],math.sqrt(testScore[0])))
    print(model.metrics_names)
    p = model.predict(X_test)
    predic = np.squeeze(np.array(p))
    predic = np.array(predic).reshape(len(predic), 1)

    predic = toMultiply.inverse_transform(predic)

    y_test = np.array(y_test).reshape(len(y_test), 1)

    y_test = toMultiply.inverse_transform(y_test)

    print(y_test)

    with open('prediction' + second + third + '.csv', 'r+') as temp:
        #content = temp.read()
        #temp.seek(0,0)
        temp.write('Atual,Previsao\n')
        #temp.write(content)
        for i in range(len(y_test)):
            temp.write(str(y_test[i][0]) + "," + str(predic[i][0]) + "\n")

def LSTM_start(first,second,third):
    df, toMultiply, scaler_ranker = load_dataset(second)
    if(first == "Freguesia"):
        janela = 24
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
    evaluate_test(second,third,model,X_test,y_test,scaler_ranker)


def main():
    total = sys.argv
    print(total)
    #primeiro argumento saber se e a longo prazo ou nao
    long_or_short = "curto"
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
