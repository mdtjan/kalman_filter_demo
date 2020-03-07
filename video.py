from collections import deque
import argparse
import random
import numpy as np
import cv2
from kalman_filter import KalmanFilter


def filter_yellow_object(img):
    yellow_bgr_lower_bound = np.array([22, 100, 100], dtype="uint8")
    yellow_bgr_upper_bound = np.array([30, 255, 255], dtype="uint8")
    mask = cv2.inRange(img, yellow_bgr_lower_bound, yellow_bgr_upper_bound)
    return mask

def find_corners(img):
    """
    Find the minimum and maximum of four coordinates (up, bottom, left, right)
    based on very simple binary thresholding
    """

    if not np.count_nonzero(img):
        return None, None, None, None
    
    x1 = np.min(np.where(mask==255)[1])
    y1 = np.min(np.where(mask==255)[0])
    x2 = np.max(np.where(mask==255)[1])
    y2 = np.max(np.where(mask==255)[0])

    return x1, y1, x2, y2

def draw_rectangle(img, x1, y1, x2, y2, color, thickness):#color,thickness):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_circle(img, x, y, r, color):
    cv2.circle(img,(x,y), r, color, -1)


def get_center_rect(x1, y1, x2, y2):
    cx = x1 + int((x2-x1)/2)
    cy = y1 + int((y2-y1)/2)
    return cx, cy


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', action='store', type=str, required=True)
    args = arg_parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.read()[0]:
        raise FileNotFoundError

    tracked_points = deque()
    
    # contrail effect
    nth_point_fade = 10
    
    random.seed(0)
    process_noise_noise_constant = random.uniform(0,1)
    measurement_noise_noise_constant = random.uniform(0,1)
    
    # Kalman Filter initialization
    state_matrix = np.zeros((4, 1))  # [x, y, delta_x, delta_y]
    estimate_cov = np.eye(state_matrix.shape[0])
    transition_matrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    process_noise_cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * process_noise_noise_constant
    measurement_state_matrix = np.zeros((2, 1))
    observation_matrix = np.array([[1,0,0,0],[0,1,0,0]])
    measurement_noise_cov = np.array([[1,0],[0,1]]) * measurement_noise_noise_constant
    
    kf = KalmanFilter(X=state_matrix, 
                      P=estimate_cov, 
                      F=transition_matrix,
                      Q=process_noise_cov, 
                      Z=measurement_state_matrix,
                      H=observation_matrix, 
                      R=measurement_noise_cov)
    
    while(cap.isOpened()):
        ret, frame = cap.read()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = filter_yellow_object(frame_hsv)

        x1, y1, x2, y2 = find_corners(mask)
        
        if not x1:
            cv2.imshow('frame', frame)
            tracked_points.clear()

            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            continue


        green_rgb = (153, 255, 51)
        red_rgb = (255,0,0)
        # light_blue_rgb = (173,216,230)

        draw_rectangle(rgb_frame, x1, y1, x2, y2, green_rgb, 1)
        
        cx, cy = get_center_rect(x1, y1, x2, y2)        
        tracked_points.append([cx, cy])

        if len(tracked_points) % nth_point_fade == 1 and len(tracked_points) > nth_point_fade:
            tracked_points.popleft()
            
        for _point in tracked_points:
            current_state_measurement = np.array([[_point[0]], [_point[1]]])
            
            tracker_point = kf.predict()
            _ = kf.correct(current_state_measurement)

            # draw_circle(rgb_frame, _point[0], _point[1], 3, red_rgb)
            draw_circle(rgb_frame, tracker_point[0], tracker_point[1], 3, red_rgb)
        
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('frame', bgr_frame)

        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()