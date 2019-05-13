import numpy as np
import matplotlib as plt
import pandas as pd
import math,time
import sys
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

#Falta apenas endireitar esta funcao

def get_dataset(dataset_name,precoOrquant, normalized = 0, file_name = None):
    col_names = ["Mes","Dia","Ano","Hora","Minutos","Dia da semana","current_speed","free_flow_speed","speed_diff","current_travel_time","free_flow_travel_time","time_diff","free_flow_distance","total_time","total_speed","diif_ranker","length_in_meters","delay_in_seconds","incidente","temperature","atmospheric_pressure","humidity","wind_speed","cloudiness","current_luminosity","no_rain","chuva_fraca","chuva","chuva_trovoada","chuva_intensa"]
    stocks = pd.read_csv(file_name, header = 0, names = col_names)
    df = pd.DataFrame(stocks)
    df.drop(df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,24]], axis = 1, inplace= True)
    #Encontrar os maximos de cada coluna
    maxQuantity = df['SomaQuantidade'].max()
    maxSomaPreco = df['SomaPrecoUni'].max()
    #normalizar todas os atributos, e por na variavel to return qual o fator de normalizaçao da ultima coluna

    toReturn = maxQuantity
    print(df)
    return (df,toReturn)

def load_dataset(arg):
    dataset_name = arg
    return get_dataset(dataset_name, 0, dataset_name + '.csv')

def load_data(df_dados,janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.as_matrix()
    tam_sequecia = janela
    res = []
    for i in range(len(mat_dados) - tam_sequecia + 1):
        res.append(mat_dados[i: i + tam_sequecia])
    res = np.array(res)
    qt_casos_treino = int(round(0.8 * res.shape[0]))
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
        print('valor: %f ----> PrevisÃ£o: %f  Diff: %f Racio: %f' % (y_test[i], predic[i],diff[i],racio[i]))
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
    print_series_prediction(y_test, predic)

def LSTM_start(arg):
    df, toMultiply = load_dataset(arg)
    janela = 1
    X_train, y_train, X_test, y_test = load_data(df, janela)
    print("X_train",X_train.shape)
    print("y_train",y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    model = build_model(janela)
    history = model.fit(X_train,y_train,batch_size=1,epochs=100,validation_split = 0.1,verbose= 1)
    save_model_json(model, arg + "_model.json")
    save_weights_hdf5(model, arg + "_model.h5")
    save_hiper_to_json(0.1,1500,1,1,arg + "_hiperparametros.json")
    print_model(model,"lstm_model.png")
    print_history_loss(history)
    trainScore = model.evaluate(X_train, y_train, verbose= 0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0],math.sqrt(trainScore[0])))
    evaluate_test(arg, model,X_test,y_test,toMultiply)

def main():
    arg = sys.argv[1]
    LSTM_start(arg)


if __name__ == "__main__":
    main()
