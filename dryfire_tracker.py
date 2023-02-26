from __future__ import print_function
import cv2
import argparse
import numpy as np
import win32api, win32con

def click(x,y):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def move(x,y):
    win32api.SetCursorPos((x,y))

boundary_threshold = 0.05
kernel_size = 5
screenw, screenh = 1280, 720
x,y,w,h = None, None, None, None
wdiff, hdiff = None, None
startcapture = 0
startclicking = 0
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
def on_change_click_on_red_dot(val):
    global startclicking
    startclicking = val
    cv2.setTrackbarPos("Enable Click", window_detection_name, val)
def on_change_track_red_dot(val):
    global startcapture
    startcapture = val
    cv2.setTrackbarPos("Enable Move", window_detection_name, val)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv2.VideoCapture(args.camera)
focus = 20
cap.set(cv2.CAP_PROP_FOCUS, focus)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
cv2.createTrackbar("Enable Click", window_detection_name, 0, 1, on_change_click_on_red_dot)
cv2.createTrackbar("Enable Move", window_detection_name, 0, 1, on_change_track_red_dot)
on_low_H_thresh_trackbar(115)
on_low_V_thresh_trackbar(185)
while True:
    
    ret, frame = cap.read()
    if frame is None:
        break
    
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([115,0,185])
    # upper_red = np.array([180, 255, 255])
    ret_val, image = cap.read()
     # Threshold the HSV image to isolate the red color
    lower_red1 = np.array([low_H, low_S, low_V])
    upper_red1 = np.array([high_H, high_S, high_V])
    mask1 = cv2.inRange(frame_HSV, lower_red1, upper_red1)

    lower_red2 = np.array([160, low_S, low_V])
    upper_red2 = np.array([180, high_S, high_V])
    mask2 = cv2.inRange(frame_HSV, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    if x is None: 
        selectedBoundary = cv2.selectROI(image)
        x,y,w,h = int(selectedBoundary[0]), int(selectedBoundary[1]), int(selectedBoundary[2]), int(selectedBoundary[3])
        print("detection area", x,y,w,h)
        wdiff = screenw / w
        hdiff = screenh / h

    cropped = mask[y:y+h, x:x+w]
    cv2.imshow('mask', cropped)
    cv2.imshow(window_detection_name, frame)

    pixels =np.where(cropped==[255]) #Pixels for the Detected
    if len(pixels[0]) > 0:
        coors =np.argwhere(cropped==[255])
        setw = -screenw + (coors[0][1] * wdiff)+5
        seth = coors[0][0] * hdiff+10
         # Calculate the boundary area based on the threshold
        boundary_w = int(w * boundary_threshold)
        boundary_h = int(h * boundary_threshold)
        if coors[0][1] > boundary_w and coors[0][1] < (w - boundary_w) and \
           coors[0][0] > boundary_h and coors[0][0] < (h - boundary_h):
            print("Pixel x,y:", coors)
            print(setw, seth)
            if startcapture == 1:
                move(int(setw), int(seth))
            if startclicking == 1:
                click(int(setw), int(seth))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    if cv2.waitKey(1) & 0xFF == ord('s'):
        selectedBoundary = cv2.selectROI(image)
        x,y,w,h = int(selectedBoundary[0]), int(selectedBoundary[1]), int(selectedBoundary[2]), int(selectedBoundary[3])
        print("detection area", x,y,w,h)
        wdiff = screenw / w
        hdiff = screenh / h

cap.release()
cv2.destroyAllWindows()