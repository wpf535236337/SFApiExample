import os
import string
import json
import cv2
from datetime import datetime
from flask import Flask, request, make_response, jsonify
import urllib
from urllib.request import urlretrieve

from process import ApiExample

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Initialize our Flask application and the PyTorch model.
app = Flask(__name__)
# keep the source dictionary order
app.config['JSON_SORT_KEYS'] = False

# establish the core process
# hard example saving: if confidence less than threshold, save the image
HE_T = 0.8


# test the api available
@app.route('/')
def hello():
    return "Hello"


@app.route('/classify_scene', methods=['POST'])
def classify_scene():
    if request.method == 'POST':
        # Initialize the data dictionary that will be returned
        res_data = {'success': True, 'error_msg': '', 'scene_name': '', 'confidence': {}}

        # get the image url
        try:
            # get the post data which is json format
            image_url = json.loads(request.get_data().decode('utf-8'))['image_url']
            # process chinese format string
            image_url = urllib.parse.quote(image_url, safe=string.printable)
        except Exception as e:
            res_data['success'] = False
            res_data['error_msg'] = 'Exception: ' + repr(e)

        # download the image
        if res_data['success']:
            try:
                image_root = os.path.join('hard_example_dataset', datetime.now().strftime("%Y%m%d"))
                os.makedirs(image_root, exist_ok=True)
                image_path = os.path.join(image_root, os.path.basename(image_url))
                # using local image to debug if the network it's not available
                # image_path = 'test_input/2020-03-29-1585423004059.jpg'
                urlretrieve(image_url, image_path)
            except Exception as e:
                res_data['success'] = False
                res_data['error_msg'] = 'Exception: ' + repr(e)

        # process the image
        if res_data['success']:
            try:
                image = cv2.imread(image_path, 1)
                prob, scene_name = scene_rec(image)
                res_data['scene_name'] = scene_name
                res_data['confidence'] = "{0:.4f}".format(prob)
                if prob > HE_T:
                    os.remove(image_path)
            except Exception as e:
                res_data['success'] = False
                res_data['error_msg'] = 'Exception: ' + repr(e)

        # return the result
        rst = make_response(jsonify(res_data))
        rst.headers['Access-Control-Allow-Origin'] = '*'

        return rst


if __name__ == '__main__':
    # select the gpu number
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    scene_rec = ApiExample()
    app.run(host="0.0.0.0", port=10090, debug=False)