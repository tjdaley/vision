import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read an image file using grayscale
# cv2.IMREAD_COLOR
# cv2.IMREAD_UNCHANGED
# cv2.IMREAD_GRAYSCALE
image = cv2.imread('ava.jpg', cv2.IMREAD_GRAYSCALE)

# Show the image. Hold the window until the user presses a key
cv2.imshow('AVA', image)
keystroke = cv2.waitKey(0) & 0xFF

# Save file if user pressed "s", otherwise just exit
if keystroke == 27: # ESC key
	cv2.destroyAllWindows()
elif keystroke == ord('s'):
	cv2.imwrite('AVA_GRAY.png', image)
	cv2.destroyAllWindows()
	
# Now try it with matplotlib
plt.imshow(image, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # Hides tick-marks
plt.show()