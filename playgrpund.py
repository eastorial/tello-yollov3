
import cv2
import argparse
import time
import numpy as np
from twilio.rest import Client
import datetime
from time import gmtime, strftime
from ftplib import FTP
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
import requests
from ip2geotools.databases.noncommercial import DbIpCity
from datetime import datetime
from pytz import timezone

from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import imutils



ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',
				help = 'path to yolo config file', default='yolov3-tiny.cfg')
ap.add_argument('-w', '--weights',
				help = 'path to yolo pre-trained weights', default='yolov3-tiny.weights')
ap.add_argument('-cl', '--classes',
				help = 'path to text file containing class names',default='coco.names')
args = ap.parse_args()

def classify_frame(net, inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			image = inputQueue.get()
			blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
			net.setInput(blob)
			detections = net.forward(['yolo_16', 'yolo_23'])
			# write the detections to the output queue
			outputQueue.put(detections)


def getTimeJakarta():
	format = "%Y-%m-%d %H:%M:%S %Z%z"
	now_utc = datetime.now(timezone('UTC'))
	now_asia = now_utc.astimezone(timezone('Asia/Jakarta'))
	return now_asia.strftime(format)

def getLocation():
	ip = requests.get('https://checkip.amazonaws.com').text.strip()
	response = DbIpCity.get(ip, api_key='free')
	lat=response.latitude
	long=response.longitude
	return "http://maps.google.com/maps?q="+str(lat)+","+str(long)


def sendEmail(toEmail,fromEmail,Subject,detail,latlong,attachmentImage):
	# Define these once; use them twice!
	strFrom = fromEmail
	strTo = toEmail

	# Create the root message and fill in the from, to, and subject headers
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = Subject
	msgRoot['From'] = strFrom
	msgRoot['To'] = strTo
	msgRoot.preamble = 'This is a multi-part message in MIME format.'

	# Encapsulate the plain and HTML versions of the message body in an
	# 'alternative' part, so message agents can decide which they want to display.
	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)

	msgText = MIMEText('This is the alternative plain text message.')
	msgAlternative.attach(msgText)

	# We reference the image in the IMG SRC attribute by the ID we give it below
	#msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>Nifty!', 'html')
	msgText = MIMEText('<b>'+detail+' <i> detected at </i>' +latlong +'</b><br><img src="cid:image1"><br>', 'html')
	msgAlternative.attach(msgText)

	# This example assumes the image is in the current directory
	fp = open(attachmentImage, 'rb')
	msgImage = MIMEImage(fp.read())
	fp.close()

	# Define the image's ID as referenced above
	msgImage.add_header('Content-ID', '<image1>')
	msgRoot.attach(msgImage)

	# Send the email (this example assumes SMTP authentication is required)

	smtp = smtplib.SMTP('smtp.zoho.com',587)
	smtp.starttls()
	smtp.login('arief@rastek.id', 'Mibmlw4e2477')
	smtp.sendmail(strFrom, strTo.split(','), msgRoot.as_string())
	smtp.quit()
	print('Mail Sent To : '+strTo)

def sendFile(fileName):
	ftp = FTP()
	ftp.set_debuglevel(2)
	ftp.connect('206.189.93.218', 21)
	ftp.login('hik59','bayem15rastek')
	ftp.cwd('/atm')

	fp = open(fileName, 'rb')
	ftp.storbinary('STOR %s' % os.path.basename(fileName), fp, 2048)
	fp.close()
	print("after upload " + fileName)


def send_wa(body_text,detail_text,phone_number):

	account_sid = 'AC93f59976e828a2a31abe40b6e6a331fa'
	auth_token = '5562e4a20995379bd09b239a4144ca40'
	client = Client(account_sid, auth_token)

	message = client.messages.create(
							  from_='whatsapp:+14155238886',
							  body=body_text +'. ' + detail_text,
							  to='whatsapp:'+phone_number
						  )

	print(message.sid)


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
	layersNames = net.getLayerNames()
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

	label = str(classes[class_id])


	color = COLORS[class_id]

	cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

	cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	conf = str(confidence)
	cv2.putText(img, conf, (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)




# Load names classes
classes = None
detectFirstTime = True
dt_st_str = None

with open(args.classes, 'r') as f:
	classes = [line.strip() for line in f.readlines()]
print(classes)

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args.weights,args.config)

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
# Define video capture for default cam
# cap = VideoStream(src=0).start()
cap =VideoStream('rtsp://admin:ipcam@reog39@10.168.1.126').start()
time.sleep(2.0)



while True:

	image = cap.read()
	# image = imutils.resize(image, width=320)
	(fH, fW) = image.shape[:2]
	Width = image.shape[1]
	Height = image.shape[0]
	#image=cv2.resize(image, (620, 480))

	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(image)

	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		detections = outputQueue.get()

	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.7
	nms_threshold = 0.4

	if detections is not None:
		for out in detections:
		#print(out.shape)
			for detection in out:
				
				#each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
				scores = detection[5:] #classes scores starts from index 5
				class_id = np.argmax(scores)
				confidence = scores[class_id]

				if confidence > 0.2:
					center_x = int(detection[0] * Width)
					center_y = int(detection[1] * Height)
					w = int(detection[2] * Width)
					h = int(detection[3] * Height)
					x = center_x - w / 2
					y = center_y - h / 2
					class_ids.append(class_id)
					confidences.append(float(confidence))
					boxes.append([x, y, w, h])

	# apply  non-maximum suppression algorithm on the bounding boxes
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

		for i in indices:
			print(i)
			i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]
			draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
			if 'person' in str(str(classes[class_id])):
				print("obeng detected at: "+getTimeJakarta())
				dt_st = datetime.now(timezone('UTC'))
				curr_min_st = dt_st.minute
				if dt_st_str is None:
					dt_st_str=curr_min_st

				dt_ended = datetime.now(timezone('UTC'))
				curr_min_ended = dt_ended.minute
				diff = abs(curr_min_ended-dt_st_str)

				print("dt_st_str : "+str(dt_st_str))
				print("curr_min_ended : "+str(curr_min_ended))
				if diff >= 1 or detectFirstTime  :
					print("will send email...")
					print("diff : "+str(diff))
					imageCopy=image.copy()
					#fileName="obeng-"+strftime("%Y-%m-%d %H:%M:%S", gmtime())+".jpg"
					fileName=str(str(classes[class_id]))+" - "+getTimeJakarta()+".jpg"
					cv2.imwrite(fileName,imageCopy)
					#sendEmail("iqbal@rastek.id,arief.djauhari@gmail.com,bambang.hardiyanto@indosatooredoo.com,bambang.hardiyanto@gmail.com","arief@rastek.id",str(str(classes[class_id]))+" detected","obeng  "+strftime("%Y-%m-%d %H:%M:%S %z", gmtime()),getLocation(),fileName)
					#sendEmail("iqbal@rastek.id,arief.djauhari@gmail.com","arief@rastek.id",str(str(classes[class_id]))+" detected","obeng  "+getTimeJakarta(),getLocation(),fileName)
					detectFirstTime=False
					dt_st_str=None
					#sendFile(os.path.basename(fileName))
					#send_wa("detected : " + str(str(classes[class_id])) +", at "+ str(datetime.datetime.now()),"image at : ftp://206.189.93.218/atm/"+fileName,"+628164853469")
					#send_wa("detected : cell phone, at "+ str(datetime.datetime.now()),"image at : ftp://206.189.93.218/atm/"+fileName,"+6281323781301")




		cv2.imshow("raV", image)

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


	# stop the timer and display FPS information


	# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()