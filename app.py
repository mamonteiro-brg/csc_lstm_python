from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import subprocess
import json


app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/da', methods=['GET'])
@cross_origin()
def process():
    print(request.args)
    option1 = request.args.get("sele1")
    option2 = request.args.get("sele2")
    option3 = request.args.get("tipo")
    option2 = option2.replace(' ', '_')
    print(option1)
    print(option2)
    print(option3)
    #subprocess.call('python LSTM_model.py ' + option3 + ' ' + option1 + ' ' + option2)
    if(option2 == "S._Vitor" and option3 == "curto"):
        print("Entrei")
        option2 = "svitor"

    csvp  = "prediction" + option2 + option3 + ".csv"

    data = pd.read_csv(csvp)

    data = data.reset_index().to_json(orient='records')

    print(data)


    #data.drop(["Atual"],axis=1, inplace=True)
    #print(data)
    #data_matrix = data.to_records()
    #print(data_matrix)
    #print(data_matrix.tolist())
    #js = {}
    #for i in data_matrix.tolist():
    #    js[i[1]] = i[2]

    #print(jsonify(js))
    #print(json.dumps(js))
    return jsonify(data)

if __name__ == '__main__':
    app.run()
    