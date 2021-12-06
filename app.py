import cv2
import numpy as np
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import json
import os
size = 416
input_size = size

# Initialize Flask application
app = Flask(__name__)

with open('./core/coco.names', 'r') as f:
        classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('./core/yolov4.cfg', './core/yolov4-tiny.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# API that returns JSON with classes found in images
@app.route('/detections/by-image-files', methods=['POST'])
def get_detections_by_image_files():
    #read image file string data
    filestr = request.files['image'].read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image_data = cv2.resize(img, (input_size, input_size))

    # create list for final response
    responses =[]
   
    classIds, scores, boxes = model.detect(image_data, confThreshold=0.55, nmsThreshold=0.325)
    
    for (classId, score, box) in zip(classIds, scores, boxes):
        try:
            responses.append({
                "class": str(classes[classId[0]]),
                "confidence": str(score),
                "box": box.tolist()
            })
        except:
            responses.append({
                "class": str(classes[classId]),
                "confidence": str(score),
                "box": box.tolist()
            })


    try:
        return Response(response=json.dumps({"response": {"detections": responses}}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)




#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=os.environ.get('PORT',8080), debug=true)
