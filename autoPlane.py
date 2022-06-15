from concurrent.futures import thread
import numpy as np
import cv2
import yaml
import time
import threading

from djitellopy import Tello
from configPlane import plane

def findObjects(classes, outputs, img):

    confThr = 0.5
    nmsThr = 0.4

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

def renderCam(tello, classes, net):
      
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   out = cv2.VideoWriter('output.avi', fourcc, 20.0, (720,  480))
   whT = 320
   
   while True :
      
      img = tello.get_frame_read().frame
      img = cv2.resize(img, (720, 480))

      blob = cv2.dnn.blobFromImage(
         img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
      net.setInput(blob)
      layers = net.getLayerNames()
      outputN = [(layers[i-1]) for i in net.getUnconnectedOutLayers()]
      outputs = net.forward(outputN)
      findObjects(classes, outputs, img)
      
      cv2.imshow("drone", img)
      
      frame = cv2.flip(img, 0)
      out.write(frame)
      
      key = cv2.waitKey(1) & 0xff
      if key == 27:  # ESC
         time.sleep(1)
         tello.streamoff()
         break
      
   out.release()
   cv2.destroyAllWindows()
   
   
   
   
def configPlane(tello, configAutoPlane):
   time.sleep(4)
   tello.takeoff()
   for i in configAutoPlane:  
      for j in configAutoPlane[i]:    
         plane(tello, j, configAutoPlane[i][j]) 

def index():
   
   with open("configAutoPlane.yaml", "r") as f:
      configAutoPlane = yaml.load(f, Loader=yaml.FullLoader)
   
   with open('coco.names', 'r') as f:
      classes = f.read().rstrip('\n').split("\n")
   
   tello = Tello()
   tello.connect()
   tello.streamon()
   
   keepRecording = True
   
   net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

   t1 = threading.Thread(target=renderCam, args=(tello, classes, net,))
   t2 = threading.Thread(target=configPlane, args=(tello, configAutoPlane,))
  
   print(tello.get_battery())
   t1.start()
   t2.start()
  
   t1.join()
   t2.join()
   
   keepRecording = False
   
index()  