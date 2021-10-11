# Importing libraries
import pickle
from flask import Flask
from flask import request
from flask import jsonify

# Parameteres

C = 1.0
n_splits = 5
input_file = f'model_C={C}.bin'

# # Loading the model

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')


@app.route('/predict_flask', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
