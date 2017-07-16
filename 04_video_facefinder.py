import numpy as np
import cv2

class Classifier():
	def __init__(self, name, color, org, cascadeclassifier):
		self.name = name
		self.color = color
		self.org = org
		self.cascade = cascadeclassifier
		self.counter = 0
		
	def nameWithCounts(self):
		return ((self.name+": {}")).format(self.counter)
		
	def incrementCounter(self):
		self.counter += 1

# Load cascade file for detecting faces		
classifiers = []
classifiers.append(Classifier("Default", (255,0,0), (0,20), cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_frontalface_default.xml')))
classifiers.append(Classifier("Alt", (255,255,0), (0,45), cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')))
classifiers.append(Classifier("Alt2", (0,255,0), (0,70), cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_frontalface_alt2.xml')))
classifiers.append(Classifier("AltTree", (0,0,255), (0,95), cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_frontalface_alt_tree.xml')))

# Not using this for now
eye_cascade  = cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_eye.xml')

# Total number of frames processed
frameCount = 0

# Connect to first (and probably only) camera
cap = cv2.VideoCapture(0)

while True:
	# Capture a frame
	ret, frame = cap.read()
	
	# Create a gray-scale version for use by the classifier
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #note reversed colors

	# Print classifier names in color that corresponds to their rectangles
	for classifier in classifiers:
		cv2.putText(frame, classifier.nameWithCounts(), classifier.org, cv2.FONT_HERSHEY_SIMPLEX, 1, classifier.color, 2)

	frameCount += 1
	
	# Run each classifier on the gray-scaled image
	for classifier in classifiers:
		
		# Look for faces
		faces = classifier.cascade.detectMultiScale(gray, 1.1, 5)

		# Draw a rectangle around each face
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), classifier.color, 2)
			#roi_gray  = gray[y:y+h, x:x+w]
			#roi_color = frame[y:y+h, x:x+w]
			classifier.incrementCounter()
		
			#eyes = eye_cascade.detectMultiScale(roi_gray)
			#eyecount = 0
			#for (ex, ey, ew, eh) in eyes:
			#	eyecount += 1
			#	cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),(0, 255, 0), 2)
			#	if eyecount == 2: break

	# Show the image. Loop until the user presses a key
	cv2.imshow('FACES', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# PRINT STATS
for classifier in classifiers:
	print((classifier.name + " {}").format((classifier.counter/frameCount)*100))