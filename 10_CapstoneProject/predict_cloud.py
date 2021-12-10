import requests

url = 'https://c1pil6vcjj.execute-api.eu-central-1.amazonaws.com/brain-tumor-test/predict'

# from test dataset
#data = {'url': 'https://raw.githubusercontent.com/sanghvirajit/ML-Zoomcamp/main/10_CapstoneProject/Data/test/yes/Y138.jpg'}

# random brain tumor picture from internet
data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/5/5f/Hirnmetastase_MRT-T1_KM.jpg'}

result = requests.post(url, json=data).json()
print(result)

if result['yes'] > 0.5:
    print("It has brain tumor")
else:
    print("It doesn't have brain tumor")