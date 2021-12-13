# Importing libraries
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

# Parameteres

input_file = 'model_xgb.bin'

# Loading the model

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('fraud_detection')

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()
    xcustomer = dv.transform([customer])
    dcustomer = xgb.DMatrix(xcustomer, feature_names=dv.get_feature_names())
    y_pred = model.predict(dcustomer)
    defualt = y_pred >= 0.5

    result = {
        'Fraud_probability': float(y_pred),
        'Fraud': bool(defualt)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
