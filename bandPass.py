#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:32:32 2018

@author: rezwan
"""
from libraries import *

def bandPass(data):
#    data = single_ppg # one video makes one peg (one for red, one for green and one  for blue)
    
    fps = 60 # Pixel 2 captured video using 60 FPS
    
    BPM_L = 40 # minimum Blood Pulse per min
    BPM_H = 500 # Max blood use per min
    
    Order = 2
    
    b, a = scipy.signal.butter(Order, [(BPM_L / 60) * (2 / fps), (BPM_H / 60) * (2 / fps)], btype='band')
    
    filtered_data = scipy.signal.filtfilt(b, a, data) # This is a filtered PPG
    
    return filtered_data

#data = [125, 128, 222, 136, 185, 125, 128, 222, 136, 185, 125, 128, 222, 136, 185, 125, 128, 222, 136, 185]
#
#print(bandPass(data))