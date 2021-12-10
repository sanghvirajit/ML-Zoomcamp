import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# random brain tumor picture from internet
data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/5/5f/Hirnmetastase_MRT-T1_KM.jpg'}

result = requests.post(url, json=data).json()
print(result)