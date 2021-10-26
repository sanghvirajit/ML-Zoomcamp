import requests
import pandas as pd
import json

url = 'http://localhost:9696/predict'

#host = 'churn-serving-env.eba-gyzkzxig.eu-central-1.elasticbeanstalk.com'
#url = f'http://{host}/predict_flask'

df = pd.read_csv("customer_test.csv")

customer_number = 1

customer = df.iloc[customer_number-1].to_json()
customer = json.loads(customer)

response = requests.post(url, json=customer).json()
print(response)

if response['defualt'] == True:
    print("Customer is not creditworthiness")
else:
    print("Customer is creditworthiness")

