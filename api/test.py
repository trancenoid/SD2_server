import base64
import json                    
import requests

api = 'http://localhost:8000/impaint'
image_file = 'test.jpg'

f1 = open(image_file, "rb")
f2 = open(image_file, "rb")

files = { 'image' : ('test.jpg', f1, 'image/jpeg'), 'mask' : ('test.jpg', f2, 'image/jpeg') } 

headers = {'Accept': 'application/json'}
  

response = requests.post(api, params = {'prompt' : "YO"}, files=files, headers=headers)

f1.close()
f2.close()


try:
    data = response.json()     
    print(data)                
except requests.exceptions.RequestException:
    print(response.text)