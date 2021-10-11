import requests

#url = 'http://localhost:9696/predict_flask'

host = 'churn-serving-env.eba-gyzkzxig.eu-central-1.elasticbeanstalk.com'
url = f'http://{host}/predict_flask'

customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print("sending promo email to customer id")
else:
    print("Not sending promo email")

