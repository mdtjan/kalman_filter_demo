import numpy as np
import cv2
from collections import deque
import argparse


def filter_yellow_object(img):
    '''
    input: BGR image
    '''
    yellow_bgr_lower_bound = np.array([22, 100, 100], dtype="uint8")
    yellow_bgr_upper_bound = np.array([30, 255, 255], dtype="uint8")
    mask = cv2.inRange(img, yellow_bgr_lower_bound, yellow_bgr_upper_bound)
    return mask

def find_corners(img):
    '''
    # very simple based on binary thresholding
    # assume perfect binary thresholding
    
    '''
    if not np.count_nonzero(img):
        return None, None, None, None
    
    x1 = np.min(np.where(mask==255)[1])
    y1 = np.min(np.where(mask==255)[0])
    x2 = np.max(np.where(mask==255)[1])
    y2 = np.max(np.where(mask==255)[0])

    return x1, y1, x2, y2

def draw_rectangle(img, x1, y1, x2, y2, color, thickness):#color,thickness):
    '''
    
    '''
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def foo():
    '''
    
    '''
    # maintain deque "contrail" effect
    # n slides
    pass

def draw_circle(img, x, y, r, color):
    # input RGB image
    cv2.circle(img,(x,y), r, color, -1)

def get_center_rect(x1, y1, x2, y2):
    cx = x1 + int((x2-x1)/2)
    cy = y1 + int((y2-y1)/2)
    return cx, cy

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid_test_kf.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = filter_yellow_object(frame_hsv)

        x1, y1, x2, y2 = find_corners(mask)
        
        if not x1:
            cv2.imshow('frame', frame)

            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            continue

        # if any of these is null, then skip (v)
        # draw the circle only when there's mask
        # maintain the circle n frames after
        # circle is placed in the middle of rectangle
        green_rgb = (153, 255, 51)
        red_rgb = (255,0,0)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        draw_rectangle(rgb_frame, x1, y1, x2, y2, green_rgb, 1)
        
        cx, cy = get_center_rect(x1, y1, x2, y2)
        draw_circle(rgb_frame, cx, cy, 3, red_rgb)
        
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('frame', bgr_frame)

        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()