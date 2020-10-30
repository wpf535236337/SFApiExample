# -*- coding: utf-8 -*
import json
import requests

if __name__ == '__main__':
    HEADERS = {'Content-Type': 'application/json;charset=UTF-8',
               'apikey': '',
               'server-name': ''}
    # images
    endpoint = 'http://0.0.0.0:10090/classify_scene'
    url_data = {"image_url": "https://www.dataquest.io/wp-content/uploads/2015/09/iss056e201262-768x512.jpg"}
    content = requests.post(url=endpoint, headers=HEADERS, data=json.dumps(url_data))
    if content.status_code == 200:
        print(content.json())