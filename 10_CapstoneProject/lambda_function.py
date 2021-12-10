import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO
from urllib import request

classes = [
    'yes'
]

interpreter = tflite.Interpreter(model_path='VGG16-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url, target_size):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 255.0
    return x

def predict(url):

    img = download_image(url, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])

    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

