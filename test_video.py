import numpy as np
import cv2
from collections import deque

cap = cv2.VideoCapture('WhatsApp Video 2020-03-06 at 00.08.05.mp4')
idx = 0
save_dir = '/home/dominikus/study/simple_tracking_kalman_filter/true_random_frames/'

with open('true_random_idx', 'r') as f:
    true_random = f.read()
true_random = true_random.split('\n')
true_random.pop(-1)
true_random = [int(i) for i in true_random]

while(cap.isOpened()):
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(28) & 0xFF == ord('q'):
        break
    
    if idx in true_random:
        cv2.imwrite(save_dir+str(idx)+'.jpg', frame)
    idx+=1
    
    

cap.release()
cv2.destroyAllWindows()



lower = np.array([22, 100, 100], dtype="uint8")
upper = np.array([30, 255, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

def find_four_corners(img):
    # very simple based on binary thresholding
    '''
    
    '''
    pass

def draw_rectangle(img):
    '''
    
    '''
    pass

def foo():
    '''
    
    '''
    # use deque to draw line
    pass
