import numpy as np
import cv2

# Connect to first (probably only) video capture device
cap = cv2.VideoCapture(0)

# Frame by frame capture to the screen.
while True:
	ret, frame = cap.read()
	
	# Frame operations (convert to grayscale)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #note reversed colors
	
	# Display the frame we captured and converted
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Clean up before we exit
cap.release()
cv2.destroyAllWindows()