#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:13:19 2018

@author: rezwan
"""


from libraries import *
from video_to_frames import *
from plot_peak_frq import *
from plot_time_series import *
from high_pass_filter import *
from FFT import *
from peakdetect import *


#### Enter the Sourse and Destination folder
src_dir = "/media/rezwan/Softwares/THESIS/RawData/LED-Broad"
dst_dir = "/media/rezwan/Softwares/THESIS/RawData/Extract-Images"
#src_dir = "/media/rezwan/Softwares/THESIS/LED-Broad"
#dst_dir = "/media/rezwan/Softwares/THESIS/Extract-Images"
dataFrameArr = []

# Enter the LED-Broad
LED_Broad_num = ["/LED-0850", "/LED-0940", "/LED-1070","/LED-NO"]
#LED_Broad_num = ["/LED-NO"]

################################
#       Extract Frames         #
################################

for broad in LED_Broad_num:
    sourceDir = src_dir + broad + "/*.mp4"
    destinationDir = dst_dir + broad
    print(destinationDir)
    
    if not os.path.exists(destinationDir):
        os.mkdir(destinationDir)
        
    vList = glob.glob(sourceDir)
    # print(vList)
    
    for i in range(len(vList)):
        vDirName = vList[i]
        head, tail = os.path.split(vDirName)
        # get frames from video
        path = video_to_frames(vDirName, destinationDir, tail)
        dataFrameArr.append(path)

print(dataFrameArr) ### Load the directory of all image's frame file.
################################
#     End Extract Frames       #
################################