import cv2
import numpy as np
import screeninfo

VID_HEIGHT = 1080
VID_WIDTH = 1920
DEFAULT_FPS = 10

SET_AUTO_RESOLUTION = True

WINDOW_NAME = "WEBCAM"

cap = cv2.VideoCapture(0)                                   # '0' -> default cam = webcam

if SET_AUTO_RESOLUTION:
    monitor = screeninfo.get_monitors()[0]                  # DEFAULT_MONITOR_PROPS
    cap.set(3, monitor.width)
    cap.set(4, monitor.height)
else:
    cap.set(3, VID_WIDTH)                                   # WIDTH
    cap.set(4, VID_HEIGHT)                                  # HEIGHT



cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)                      # FPS
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)                       # BRIGHTNESS


print(cap.get(cv2.CAP_PROP_FPS))


# MAIN LOOP
while cap.isOpened():
    try:
        success, img = cap.read()
        __img_cpy = img.copy()

        # __hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # __gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        print("QUITTING: KEY_INTERRUPTION .. ")
        break

cap.release()
cv2.destroyAllWindows()

