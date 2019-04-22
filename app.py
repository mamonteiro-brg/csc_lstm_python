from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/da', methods=['GET'])
@cross_origin()
def process():
    print(request.args)
    option1 = request.args.get("sele1")
    option2 = request.args.get("sele2")
    print(option1)
    print(option2)
    return jsonify({'name': 'Consegui a cena'})

if __name__ == '__main__':
    app.run()
    