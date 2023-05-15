import cv2
import numpy as np

# reading image
img = cv2.imread('test1.png')

# converting image into hsv image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Defining lower and upper bound HSV values
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

lower_red = np.array([-10, 100, 100])
upper_red = np.array([10, 255, 255])

lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Defining mask for detecting color
mask1 = cv2.inRange(hsv, lower_green, upper_green)
mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
mask3 = cv2.inRange(hsv, lower_red, upper_red)
mask4 = cv2.inRange(hsv, lower_yellow, upper_yellow)

# setting threshold of hsv image
_, threshold = cv2.threshold(mask1+mask2+mask3+mask4, 250, 255, cv2.THRESH_BINARY)

# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
# list for storing names of shapes
for contour in contours:

    # here we are ignoring first counter because find contour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 0), 3)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        # putting shape color

        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            if (aspectRatio >= 0.95) and (aspectRatio <= 1.05):
                cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
            else:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        else:
            cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# displaying the image after drawing contours
cv2.imshow('Detected shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()