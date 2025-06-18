#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 00:10:54 2018

@author: rezwan
"""
from libraries import *
#from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt

import padasip as pa

#%matplotlib inline 
plt.style.use('ggplot') # nicer plots
np.random.seed(52102) # always use the same random seed to make results comparable
#%config InlineBackend.print_figure_kwargs = {}

def plot_time_series(channel_mean, color = "r", label="Channel"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_time_series_n_save_0850(channel_mean, color = "r", label="Channel", ID="_", LED_Broad="_"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Red_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Green_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_green.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Blue_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_blue.png", dpi = 100)
    plt.show()
    
def plot_1st_ppg(channel_mean, color = "r", label="Channel", ID="_", LED_Broad="_"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/PPG_1st_wave/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Green_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_green.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Blue_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_blue.png", dpi = 100)
    plt.show()
    
def plot_time_series_n_save_0940(channel_mean, color = "r", label="Channel", ID="_", LED_Broad="_"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0940/Red_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0940/Green_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_green.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0940/Blue_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_blue.png", dpi = 100)
    plt.show()
    
def plot_time_series_n_save_1070(channel_mean, color = "r", label="Channel", ID="_", LED_Broad="_"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-1070/Red_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-1070/Green_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_green.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-1070/Blue_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_blue.png", dpi = 100)
    plt.show()
    
def plot_time_series_n_save_NO(channel_mean, color = "r", label="Channel", ID="_", LED_Broad="_"):
    plt.figure(figsize=(10,6))
    plt.plot(channel_mean, color, linewidth=1, label=label)
    
    plt.xlim(0, len(channel_mean))
    plt.legend()
    plt.tight_layout()
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-NO/Red_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-NO/Green_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_green.png", dpi = 100)
#    plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-NO/Blue_PPG_Clean/"+str(ID)+"_"+str(LED_Broad)+"_blue.png", dpi = 100)
    plt.show()
    
def plot_all_chan(red, green, blue):
#        print(len(self.mean_fitness))
#        print(len(self.best_fitness))
    
    plt.figure(figsize=(8,5))
    plt.plot(red, 'r', linewidth=1, label="Red")
    plt.plot(green, 'g', linewidth=1, label="Green")
    plt.plot(blue, 'b', linewidth=1, label="Blue")
    
    plt.xlabel("Number of Frames")
    plt.ylabel("Intensity")
    plt.xlim(0, len(red))
#    plt.title("Improvement in feture set fitness over time")
    plt.legend()
    plt.tight_layout()
    plt.show()