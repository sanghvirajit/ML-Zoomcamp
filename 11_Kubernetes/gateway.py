import os
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor
from proto import np_to_protobuf

from flask import Flask, request, jsonify

#host = 'localhost:8500'
server = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def prepare_request(data):

    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(data))

    return pb_request

def prepare_response(pb_response):

    resposnse = pb_response.outputs['dense_7'].float_val
    result = {c: p for c, p in zip(classes, resposnse)}

    return result

def apply_model(url):

    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    pred = prepare_response(pb_response)

    return pred

app = Flask('clothing-model')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.get_json()
    result = apply_model(url['url'])
    return jsonify(result)

if __name__ == '__main__':

    url = 'http://bit.ly/mlbookcamp-pants'
    response = apply_model(url)
    print(response)

    #app.run(debug=True, host='0.0.0.0', port=9696)






