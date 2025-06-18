#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 22:32:04 2018

@author: rezwan
"""
from libraries import *

def video_to_frames(video_filename, destinationDir, tail):
    """Extract frames from video"""
    
    path = destinationDir + "/" + tail[:-4]
    print(path)
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0, 
                         round(video_length * 0.25), 
                         round(video_length * 0.5),
                         round(video_length * 0.75),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()

            cv2.imwrite(os.path.join(path, str(count) + '.jpg'), image)

            count += 1
            
#             if count == 300:
#                 break
                
    return path
