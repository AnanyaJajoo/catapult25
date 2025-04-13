import requests

json = {
    "query": "street",
    "k": 5
}

data = requests.post(url="http://127.0.0.1:5000/search", json=json)
print(data.json())