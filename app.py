from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os
import subprocess
import LSTM_model

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
    return jsonify({'tipo': option3})

if __name__ == '__main__':
    app.run()
    