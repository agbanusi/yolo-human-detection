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
    filestr = request.files['image']
    #convert string data to numpy array
    npimg = np.fromfile(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (416, 416))
    # create list for final response
    responses =[]
   
    classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.2)
    print(classIds, scores)
    
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

@app.route('/detections/by-alternate', methods=['POST'])
def get_detections_by_alternate():
    filestr = request.files['image']
    #convert string data to numpy array
    npimg = np.fromfile(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),[0,0,0],crop=False)
    net.setInput(blob)
    #layerNames = net.getLayerNames()
    #outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputNames = net.getUnconnectedOutLayersNames()  
    outputs = net.forward(outputNames)
    hT, wT, cT = img.shape
    responses =[]

    print(outputs)

    for output in outputs:
            for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > 0.5:
                            w,h = int(det[2]* wT), int(det[3]*hT)
                            x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                            responses.append({
                                "class": classId,
                                "confidence": str(confidence),
                                "box": ([x,y,w,h]).tolist()
                            })

    try:
        return Response(response=json.dumps({"response": {"detections": responses}}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)


#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=os.environ.get('PORT',8080), debug=true)
