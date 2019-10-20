from flask import Flask, request, Response
import numpy as np
from imutils.object_detection import non_max_suppression
import cv2
import imutils
import json

# Initialize the Flask application
app = Flask(__name__)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route('/test',methods=['GET'])
def test():
  return Response(response='Hello world...')


# route http posts to this method
@app.route('/api/facedetect', methods=['POST'])
def uploadImg():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #image = cv2.imread(imagePath)
    image = imutils.resize(img, width=min(400, img.shape[1]))
    orig = image.copy()
  
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
      padding=(8, 8), scale=1.05)
  
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
      cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
  
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
  
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
      cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
  
    # show some information on the number of bounding boxes
    # filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(
    #   filename, len(rects), len(pick)))
  
    # show the output images
    cv2.imwrite('detected.jpg',image)
    cv2.waitKey(0)

    response = {}
    response['people'] = len(pick)

    return Response(response=json.dumps(response), status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)