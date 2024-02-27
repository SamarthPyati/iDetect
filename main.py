import cv2
import numpy as np
import screeninfo
import pytesseract
from numba import njit

VID_HEIGHT = 1080
VID_WIDTH = 1920
DEFAULT_FPS = 60
# SETTING FUNCTIONS
SET_AUTO_RESOLUTION = True
RECOGNISE_TEXT = False


@njit
def recognise_text(_image):
    ''' Capture the text from image '''
    return pytesseract.image_to_string(image=_image)


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

        __hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        __gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow(WINDOW_NAME, __gray_scaled)
        if RECOGNISE_TEXT:
            print(pytesseract.image_to_string(image=__img_cpy))

        if cv2.waitKey(1) & 0xFF == ord('q'):               # 'q' to quit
            break

    except KeyboardInterrupt:
        print("QUITTING: KEY_INTERRUPTION ...")
        break

cap.release()
cv2.destroyAllWindows()
