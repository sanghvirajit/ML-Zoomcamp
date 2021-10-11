import requests

url = 'http://localhost:9696/predict_flask'

customer = {
    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 10
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print("sending promo email to customer id")
else:
    print("Not sending promo email")

