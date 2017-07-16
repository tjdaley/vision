import numpy as np
import cv2

# Load cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('.\cascades\opencv\data\haarcascades\haarcascade_eye.xml')

# Connect to first (and probably only) camera
cap = cv2.VideoCapture(0)

# Capture a frame
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #note reversed colors

# Look for faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print ("Found", len(faces), "faces")

# Draw a rectangle around each face
for (x, y, w, h) in faces:
	cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
	roi_gray  = gray[y:y+h, x:x+w]
	roi_color = frame[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),(0, 255, 0), 2)

# Show the image. Hold the window until the user presses a key
cv2.imshow('FACES', frame)
keystroke = cv2.waitKey(0) & 0xFF

# Save file if user pressed "s", otherwise just exit
if keystroke == 27: # ESC key
	cv2.destroyAllWindows()
elif keystroke == ord('s'):
	cv2.imwrite('FACES_GRAY.png', frame)
	cv2.destroyAllWindows()