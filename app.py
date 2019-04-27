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
    subprocess.call('python LSTM_model.py ' + option3 + ' ' + option1 + ' ' + option2)
    data = pd.read_csv("predictiontestPantspreco.csv")
    print(data)
    data.drop(["Atual"],axis=1, inplace=True)
    print(data)
    data_matrix = data.to_records()
    print(data_matrix)

    js = {}
    for i in data_matrix.tolist():
        js[i[0]] = i[1]

    print(json.dumps(js))
    return jsonify(js)

if __name__ == '__main__':
    app.run()
    