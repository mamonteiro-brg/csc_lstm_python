import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
import sys
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM
import json
from sklearn.preprocessing import MinMaxScaler


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

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Funcao para construcao do modelo
def build_model(janela):
    model = Sequential()

    model.add(LSTM(128, input_shape=(janela, 23), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(64, input_shape=(janela, 23), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32, input_shape=(janela, 23), return_sequences=False))

    model.add(Dense(8, activation='relu', kernel_initializer="uniform"))
    model.add(Dense(1, activation='linear', kernel_initializer="uniform"))

    # adam = optimizers.adam(lr=0.0001)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def load_data(df_dados,janela):

    print("Cenas Dados")
    print(len(df_dados))
    print(("##########################"))
    # mas a quantos registos isso equivale? Como fazer para tratar um numero de registos correcto?
    # 28*24*numero freguesia
    # Dataset Total (13002, 17)

    tam_sequecia = janela
    res = []
    for i in range(len(df_dados) - tam_sequecia + 1):
        res.append(df_dados[i: i + tam_sequecia])
        #print(res)
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

def print_series_prediction(y_test,predic,scaler):
    diff = []
    racio = []

    predic = np.array(predic).reshape(len(predic), 1)

    predic = scaler.inverse_transform(predic)

    y_test = np.array(y_test).reshape(len(y_test), 1)

    y_test = scaler.inverse_transform(y_test)

    for i in range(len(y_test)):
        racio.append((y_test[i]/predic[i])-1)
        diff.append(abs(y_test[i]-predic[i]))
        print('valor: %f ----> Previsao: %f  Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))

    print("prediction ")
    #print(predic)
    print("-------------------------------------")

    plt.plot(y_test,color = 'blue', label = 'y_test')
    plt.plot(predic, color = 'red', label = 'prediction')
    #plt.plot(diff, color = 'green', label = 'diff')
    #plt.plot(racio, color = 'yellow', label = 'racio')
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
    print('Test Socre: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    print(model.metrics_names)
    #print(X_test)
    p = model.predict(X_test)
    #print(model)
    predic = np.squeeze(np.asarray(p))
    print("prediction ")
    #print(predic)
    print("-------------------------------------")

    print_series_prediction(y_test, predic,toMultiply)

def LSTM_start(arg):
    df, toMultiply, scaler_ranker = load_dataset(arg)
    janela = 24
    print("Dataset Total", df.shape)
    X_train, y_train, X_test, y_test = load_data(df,janela)
    print("X_train",X_train.shape)
    print("y_train",y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)


    # Train a new classifier using the best parameters found by the grid search
    model = build_model(janela)
    history = model.fit(X_train,y_train,batch_size=168,epochs=150,validation_split = 0.1,verbose= 1)

    save_model_json(model, arg + "curto" + "_model.json")
    save_weights_hdf5(model, arg + "curto" + "_model_weights.h5")
    save_hiper_to_json(0.1,1500,1,1,arg + "curto" + "_hiperparametros.json")
    #print_model(model,"lstm_model.png")
    print_history_loss(history)
    trainScore = model.evaluate(X_train, y_train, verbose= 0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0],math.sqrt(trainScore[0])))
    evaluate_test(arg, model,X_test,y_test,scaler_ranker)

def main():
    #arg = sys.argv[1]
    #print(arg)
    LSTM_start('svitor')
    #f.close()


if __name__ == "__main__":
    main()