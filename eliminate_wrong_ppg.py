#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:41:15 2019

@author: rezwan
"""
from libraries import *
from plot_time_series import *
from bandPass import *
from peakdetection import peakdet
from feature_PPG_N_Back_Clip import *
from max_first_three import *
from params import *
from feature import extract_ppg45, extract_svri

                    ######################################
                    #          /LED-0850                #
                    #####################################



''' Eliminate the wrong PPG cycle. '''
df_0850_hemo = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0850_Red_Hemo.csv")
df_0850_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0850_Red_Gl.csv")
df_0850_hemo_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0850_Red_merge.csv")

## Load the text file where put the wrong ID
with open('/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850.txt', 'r') as f:
    lines = f.readlines()
    
junk_list_0850_red = []
for e in lines:
    for n in e.split(","):
        junk_list_0850_red.append(int(n))
    
print(len(junk_list_0850_red))


## Eliminate the Id from the dataframe that indicates junk_list_0850_red
##### Hemoglobin
print(df_0850_hemo.describe())
print(df_0850_hemo.shape)
print(df_0850_hemo['ID'])
print(len(df_0850_hemo['ID']))

df_0850_hemo_new = df_0850_hemo[~df_0850_hemo['ID'].isin(junk_list_0850_red)]

print(df_0850_hemo_new.describe())
print(df_0850_hemo_new.shape)
print(df_0850_hemo_new['ID'])
print(len(df_0850_hemo_new['ID']))

df_0850_hemo_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Final_dataset/df_0850_hemo.csv")


##### Glucose
print(df_0850_gl.describe())
print(df_0850_gl.shape)
print(df_0850_gl['ID'])
print(len(df_0850_gl['ID']))

df_0850_gl_new = df_0850_gl[~df_0850_gl['ID'].isin(junk_list_0850_red)]

print(df_0850_gl_new.describe())
print(df_0850_gl_new.shape)
print(df_0850_gl_new['ID'])
print(len(df_0850_gl_new['ID']))

df_0850_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Final_dataset/df_0850_gl.csv")

##### Hemo and Glucose
print(df_0850_hemo_gl.describe())
print(df_0850_hemo_gl.shape)
print(df_0850_hemo_gl['ID'])
print(len(df_0850_hemo_gl['ID']))

df_0850_hemo_gl_new = df_0850_hemo_gl[~df_0850_hemo_gl['ID'].isin(junk_list_0850_red)]

print(df_0850_hemo_gl_new.describe())
print(df_0850_hemo_gl_new.shape)
print(df_0850_hemo_gl_new['ID'])
print(len(df_0850_hemo_gl_new['ID']))

df_0850_hemo_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Final_dataset/df_0850_hemo_gl.csv")



                    ######################################
                    #          /LED-0940                #
                    #####################################


''' Eliminate the wrong PPG cycle. '''
df_0940_hemo = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0940_Red_Hemo.csv")
df_0940_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0940_Red_Gl.csv")
df_0940_hemo_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_0940_Red_merge.csv")

## Load the text file where put the wrong ID
with open('/media/rezwan/Softwares/THESIS/PPG_Img/LED-0940.txt', 'r') as f:
    lines = f.readlines()
    
junk_list_0940_red = []
for e in lines:
    for n in e.split(","):
        junk_list_0940_red.append(int(n))
    
print(len(junk_list_0940_red))


## Eliminate the Id from the dataframe that indicates junk_list_0850_red
##### Hemoglobin
print(df_0940_hemo.describe())
print(df_0940_hemo.shape)
print(df_0940_hemo['ID'])
print(len(df_0940_hemo['ID']))

df_0940_hemo_new = df_0940_hemo[~df_0940_hemo['ID'].isin(junk_list_0940_red)]

print(df_0940_hemo_new.describe())
print(df_0940_hemo_new.shape)
print(df_0940_hemo_new['ID'])
print(len(df_0940_hemo_new['ID']))

df_0940_hemo_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Final_dataset/df_0940_hemo.csv")


##### Glucose
print(df_0940_gl.describe())
print(df_0940_gl.shape)
print(df_0940_gl['ID'])
print(len(df_0940_gl['ID']))

df_0940_gl_new = df_0940_gl[~df_0940_gl['ID'].isin(junk_list_0940_red)]

print(df_0940_gl_new.describe())
print(df_0940_gl_new.shape)
print(df_0940_gl_new['ID'])
print(len(df_0940_gl_new['ID']))

df_0940_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Final_dataset/df_0940_gl.csv")


##### Hemo and Glucose
print(df_0940_hemo_gl.describe())
print(df_0940_hemo_gl.shape)
print(df_0940_hemo_gl['ID'])
print(len(df_0940_hemo_gl['ID']))

df_0940_hemo_gl_new = df_0940_hemo_gl[~df_0940_hemo_gl['ID'].isin(junk_list_0940_red)]

print(df_0940_hemo_gl_new.describe())
print(df_0940_hemo_gl_new.shape)
print(df_0940_hemo_gl_new['ID'])
print(len(df_0940_hemo_gl_new['ID']))

df_0940_hemo_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0940_PPG_CSV/Final_dataset/df_0940_hemo_gl.csv")




                    ######################################
                    #          /LED-1070                #
                    #####################################


''' Eliminate the wrong PPG cycle. '''
df_1070_hemo = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_1070_Red_Hemo.csv")
df_1070_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_1070_Red_Gl.csv")
df_1070_hemo_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_1070_Red_merge.csv")

## Load the text file where put the wrong ID
with open('/media/rezwan/Softwares/THESIS/PPG_Img/LED-1070.txt', 'r') as f:
    lines = f.readlines()
    
junk_list_1070_red = []
for e in lines:
    for n in e.split(","):
        junk_list_1070_red.append(int(n))
    
print(len(junk_list_1070_red))


## Eliminate the Id from the dataframe that indicates junk_list_0850_red
##### Hemoglobin
print(df_1070_hemo.describe())
print(df_1070_hemo.shape)
print(df_1070_hemo['ID'])
print(len(df_1070_hemo['ID']))

df_1070_hemo_new = df_1070_hemo[~df_1070_hemo['ID'].isin(junk_list_1070_red)]

print(df_1070_hemo_new.describe())
print(df_1070_hemo_new.shape)
print(df_1070_hemo_new['ID'])
print(len(df_1070_hemo_new['ID']))

df_1070_hemo_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Final_dataset/df_1070_hemo.csv")


##### Glucose
print(df_1070_gl.describe())
print(df_1070_gl.shape)
print(df_1070_gl['ID'])
print(len(df_1070_gl['ID']))

df_1070_gl_new = df_1070_gl[~df_1070_gl['ID'].isin(junk_list_1070_red)]

print(df_1070_gl_new.describe())
print(df_1070_gl_new.shape)
print(df_1070_gl_new['ID'])
print(len(df_1070_gl_new['ID']))

df_1070_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Final_dataset/df_1070_gl.csv")


##### Hemo and Glucose
print(df_1070_hemo_gl.describe())
print(df_1070_hemo_gl.shape)
print(df_1070_hemo_gl['ID'])
print(len(df_1070_hemo_gl['ID']))

df_1070_hemo_gl_new = df_1070_hemo_gl[~df_1070_hemo_gl['ID'].isin(junk_list_1070_red)]

print(df_1070_hemo_gl_new.describe())
print(df_1070_hemo_gl_new.shape)
print(df_1070_hemo_gl_new['ID'])
print(len(df_1070_hemo_gl_new['ID']))

df_1070_hemo_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-1070_PPG_CSV/Final_dataset/df_1070_hemo_gl.csv")




                    ######################################
                    #          /LED-NO                #
                    #####################################

''' Eliminate the wrong PPG cycle. '''
df_NO_hemo = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_NO_Red_Hemo.csv")
df_NO_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_NO_Red_Gl.csv")
df_NO_hemo_gl = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Preprocess_Dataset/Pre_df_PPG_45_feat_NO_Red_merge.csv")

## Load the text file where put the wrong ID
with open('/media/rezwan/Softwares/THESIS/PPG_Img/LED-NO.txt', 'r') as f:
    lines = f.readlines()
    
junk_list_NO_red = []
for e in lines:
    for n in e.split(","):
        junk_list_NO_red.append(int(n))
    
print(len(junk_list_NO_red))


## Eliminate the Id from the dataframe that indicates junk_list_0850_red
##### Hemoglobin
print(df_NO_hemo.describe())
print(df_NO_hemo.shape)
print(df_NO_hemo['ID'])
print(len(df_NO_hemo['ID']))

df_NO_hemo_new = df_NO_hemo[~df_NO_hemo['ID'].isin(junk_list_NO_red)]

print(df_NO_hemo_new.describe())
print(df_NO_hemo_new.shape)
print(df_NO_hemo_new['ID'])
print(len(df_NO_hemo_new['ID']))

df_NO_hemo_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Final_dataset/df_NO_hemo.csv")


##### Glucose
print(df_NO_gl.describe())
print(df_NO_gl.shape)
print(df_NO_gl['ID'])
print(len(df_NO_gl['ID']))

df_NO_gl_new = df_NO_gl[~df_NO_gl['ID'].isin(junk_list_NO_red)]

print(df_NO_gl_new.describe())
print(df_NO_gl_new.shape)
print(df_NO_gl_new['ID'])
print(len(df_NO_gl_new['ID']))

df_NO_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Final_dataset/df_NO_gl.csv")


##### Hemo and Glucose
print(df_NO_hemo_gl.describe())
print(df_NO_hemo_gl.shape)
print(df_NO_hemo_gl['ID'])
print(len(df_NO_hemo_gl['ID']))

df_NO_hemo_gl_new = df_NO_hemo_gl[~df_NO_hemo_gl['ID'].isin(junk_list_NO_red)]

print(df_NO_hemo_gl_new.describe())
print(df_NO_hemo_gl_new.shape)
print(df_NO_hemo_gl_new['ID'])
print(len(df_NO_hemo_gl_new['ID']))

df_NO_hemo_gl_new.to_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Final_dataset/df_NO_hemo_gl.csv")
