from urllib import response
import requests
import json

## Creates a new entry in the catalog
new_seed = {'seed':'bad dragon'}

#end_point = "host/seed/new"
end_point = ""
response = requests.post(end_point,json=new_seed)


textGenerated = json.loads(response.text)
print(textGenerated['text-generated'])