import numpy as np
from djitellopy import Tello
import cv2
import math
import time

showCap = True
classes = []
whT = 320
confThr = 0.5
nmsThr = 0.4

with open('coco.names', 'r') as f:
    classes = f.read().rstrip('\n').split("\n")

net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    wT, hT, _ = img.shape
    bbox = []
    classIds = []
    confs = []
    for out in outputs:
        for det in out:
            scores = det[5:]
            classId = np.argmax(scores)
            confindence = scores[classId]
            if confindence > confThr:
                # mean-The Object is detected
                # process
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confindence))

    indexes = cv2.dnn.NMSBoxes(bbox, confs, confThr, nmsThr)
    for i in range(len(bbox)):
        if i in indexes:
            x, y, w, h = bbox[i]
            label = str(classes[classIds[i]])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img, label, (x, y+30),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)


tello = Tello()
tello.connect()

# print(tello.get_battery())
# exit(0)

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()

while True:

    if showCap:
        img = tello.get_frame_read().frame
        img = cv2.resize(img, (720, 480))
    else:
        img = cv2.imread('1.jpg')

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers = net.getLayerNames()
    outputN = [(layers[i-1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputN)
    findObjects(outputs, img)

    drone_cam = frame_read.frame
    # cv2.imshow("drone", drone_cam)
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESCrr
        break
    elif key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)

tello.land()
