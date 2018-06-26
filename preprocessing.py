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
def show_video_loop():
    pointer = 0
    video = get_video_file(DRONE_CAPTURES[pointer])
    ret, image_np = video.read()
    image_np = cv2.bilateralFilter(image_np,9,75,75)
    prvs = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image_np)
    hsv[...,1] = 255
    while True:
        ret, image_np = video.read()
        print('Image')
        print(image_np.shape)
        if not ret:
            pointer += 1
            if pointer >= len(DRONE_CAPTURES):
              pointer = 0
            video = get_video_file(DRONE_CAPTURES[pointer])
            continue
        
        # image_np = cv2.bilateralFilter(image_np,9,75,75)
        nxt = convert_BGR2GRAY(image_np)
        flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print('Flow')
        print(flow.shape)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        print(mag.shape)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # print(get_frame_mean(image_np))

        cv2.imshow('frame',image_np)
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
    show_video_loop()
    # loop_one_video()

if __name__ == '__main__':
    main(sys.argv[1:])