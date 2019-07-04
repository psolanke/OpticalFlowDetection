import cv2
import numpy as np
import os
import sys
from PIL import Image
import time

DRONE_CAPTURES_ROOTDIR = '/media/a3s/New Volume/Youtube_Videos/DroneCaptures/'
DRONE_CAPTURES = ['DroneShot1.mp4',
                    'DroneShot2.mp4',
                    'DroneShot3.mp4',
                    'DroneShot4.mp4',
                    'DroneShot5.mp4',
                    'DroneShot6.mp4']

def get_video_file(filename):
    full_path = os.path.join(DRONE_CAPTURES_ROOTDIR, filename)
    cap = cv2.VideoCapture(full_path)
    return cap

def convert_BGR2GRAY(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_frame_mean(frame):
    return frame.mean()
# People_Walking_1.mkv

def show_video_loop_lk():
    pointer = 0
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    video = get_video_file(DRONE_CAPTURES[pointer])
    ret, image_np = video.read()
    old_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(image_np)
    while True:
        ret, image_np = video.read()
        if not ret:
            pointer += 1
            if pointer >= len(DRONE_CAPTURES):
              pointer = 0
            video = get_video_file(DRONE_CAPTURES[pointer])
            continue
        
        frame_gray = convert_BGR2GRAY(image_np)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        print(st)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            image_np = cv2.circle(image_np,(a,b),5,color[i].tolist(),-1)
        bgr = cv2.add(image_np,mask)

        cv2.imshow('frame',bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        time.sleep(0.01)

def show_video_loop_farneback():
    pointer = 0
    video = get_video_file(DRONE_CAPTURES[pointer])
    ret, image_np = video.read()
    image_np = cv2.bilateralFilter(image_np,9,75,75)
    prvs = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image_np)
    hsv[...,1] = 255
    while True:
        ret, image_np = video.read()
        if not ret:
            pointer += 1
            if pointer >= len(DRONE_CAPTURES):
              pointer = 0
            video = get_video_file(DRONE_CAPTURES[pointer])
            continue
        
        # image_np = cv2.bilateralFilter(image_np,9,75,75)
        nxt = convert_BGR2GRAY(image_np)
        flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        print(mag)
        print(np.mean(mag))
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        mag = np.subtract(mag, np.median(mag))
        print(mag)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr1 = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # print(get_frame_mean(image_np))

        cv2.imshow('frame',bgr)
        cv2.imshow('frame1', bgr1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        prvs = nxt

def loop_one_video():
    filename = '/media/a3s/New Volume/Youtube_Videos/pcroppedFile.mkv'
    video = cv2.VideoCapture(filename)
    ret, image_np = video.read()
    image_np = cv2.resize(image_np,(900,700))
    image_np = cv2.bilateralFilter(image_np,9,75,75)
    prvs = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image_np)
    hsv[...,1] = 255
    while True:
        ret, image_np = video.read()
        image_np = cv2.resize(image_np,(900,700))
        image_np = cv2.bilateralFilter(image_np,9,75,75)
        if not ret:
            video = cv2.VideoCapture(filename)
            continue
        nxt = convert_BGR2GRAY(image_np)
        flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # print(get_frame_mean(image_np))

        cv2.imshow('frame',bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        prvs = nxt

def main(args):

    # video = get_video_file(DRONE_CAPTURES[0])
    # print(type(video))
    # video = convert_video_BGR2GRAY(video)
    # while True:
    #     ret, image_np = video.read()
    show_video_loop_lk()
    # loop_one_video()

if __name__ == '__main__':
    main(sys.argv[1:])