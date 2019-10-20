from flask import Flask, request, Response
import numpy as np
from imutils.object_detection import non_max_suppression
import cv2
import imutils
import json

from mtcnn.mtcnn import MTCNN

model = MTCNN()

# Initialize the Flask application
app = Flask(__name__)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/test',methods=['GET'])
def test():
  return Response(response='Hello world...')

def detectFacesCNN(img):
  faces = model.detect_faces(img)

  for result in faces:
    x, y, w, h = result['box']
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

  cv2.imwrite('faces_detected_cnn.jpg',img)

  return len(faces)

def detectFaces(img):
  #gray = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
  # Detect faces
  faces = face_cascade.detectMultiScale(img, 1.5, 4)
  # Draw rectangle around the faces
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

  cv2.imwrite('faces_detected.jpg',img)

  return len(faces)

def detectPedestrians(img):
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

  cv2.imwrite('pedestrian_detected.jpg',image)
  
  return len(pick)


# route http posts to this method
@app.route('/api/facedetect', methods=['POST'])
def uploadImg():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = detectFacesCNN(img)
    pedestrians = detectPedestrians(img)

    response = {}
    response['people'] = max(faces,pedestrians)

    response['metadata'] = {}
    response['metadata']['faces'] = faces
    response['metadata']['pedestrians'] = pedestrians

    diff = abs(faces-pedestrians)

    response['description'] = 'People is calculated by the higher number between faces and pedestrians.'

    return Response(response=json.dumps(response), status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)