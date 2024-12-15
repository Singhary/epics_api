import requests
import json

url = 'http://127.0.0.1:8000/weather-prediction/gabba'  

# Make the GET request
response = requests.get(url)
json_respone=json.dumps(response.json(), indent=4)

print(json_respone)
