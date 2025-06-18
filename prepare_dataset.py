#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:06:47 2019

@author: rezwan
"""

from libraries import *


""" 
Preprocess the dataset for LED-0850_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

#df = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)
#df.to_csv('CSV_Files/LED-0850_PPG_CSV/PPG1_45_feat_0850_Red.csv')      

''' Merge PPG1_45_feat_0850_Red and Clinical data '''
import numpy as np
import pandas as pd
df_Clinical = pd.read_csv("./Shared AmaderGram Data 2018-DataSheet - Clinical data.csv")

df_PPG_45_feat_0850_Red = pd.read_csv('/media/rezwan/Study/undergraduate-thesis-non-invasive-hemoglobin-measurement/CSV_Files/LED-0850_PPG_CSV/Final_dataset/PPG1_45_feat_0850_Red_merge.csv')

df_PPG_45_feat_0850_Red_merge = pd.merge(df_PPG_45_feat_0850_Red, df_Clinical, on='ID', how='outer')

col_list = ['ID', 'Age', 'Sex', 'Systolic_peak(x)', 'Diastolic_peak(y)',
       'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x',
       'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)',
       'Dicrotic_notch_time(t3)',
       'Time_between_systolic_and_diastolic_peaks(∆T)',
       'Time_between_half_systolic_peak_points(w)',
       'Inflection_point_area_ratio(A2/A1)',
       'Systolic_peak_rising_slope(t1/x)',
       'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi',
       't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2',
       '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi',
       'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi',
       '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)',
       'Fundamental_component_magnitude(|sbase|)',
       '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)',
       '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)',
       'Stress-induced_vascular_response_index(sVRI)',  
       'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM']
print(len(col_list))

df_PPG_45_feat_0850_Red_merge = df_PPG_45_feat_0850_Red_merge[col_list]

dict = {'Sex':{'Male':1, 'Female':0}}      # label = column name
df_PPG_45_feat_0850_Red_merge.replace(dict,inplace = True) 

##-------- Create CSV
df_PPG_45_feat_0850_Red_merge.to_csv('./LED-0850_7_lbs.csv')


############################################# LED - 0940 ################################

""" 
Preprocess the dataset for LED-0940_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

#df = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)
#df.to_csv('CSV_Files/LED-0850_PPG_CSV/PPG1_45_feat_0850_Red.csv')      

''' Merge PPG1_45_feat_0850_Red and Clinical data '''
import numpy as np
import pandas as pd
df_Clinical = pd.read_csv("./Shared AmaderGram Data 2018-DataSheet - Clinical data.csv")

df_PPG_45_feat_0850_Red = pd.read_csv('/media/rezwan/Study/undergraduate-thesis-non-invasive-hemoglobin-measurement/CSV_Files/LED-0940_PPG_CSV/Final_dataset/PPG1_45_feat_0940_Red_merge.csv')

df_PPG_45_feat_0850_Red_merge = pd.merge(df_PPG_45_feat_0850_Red, df_Clinical, on='ID', how='outer')

col_list = ['ID', 'Age', 'Sex', 'Systolic_peak(x)', 'Diastolic_peak(y)',
       'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x',
       'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)',
       'Dicrotic_notch_time(t3)',
       'Time_between_systolic_and_diastolic_peaks(∆T)',
       'Time_between_half_systolic_peak_points(w)',
       'Inflection_point_area_ratio(A2/A1)',
       'Systolic_peak_rising_slope(t1/x)',
       'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi',
       't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2',
       '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi',
       'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi',
       '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)',
       'Fundamental_component_magnitude(|sbase|)',
       '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)',
       '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)',
       'Stress-induced_vascular_response_index(sVRI)',  
       'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM']
print(len(col_list))

df_PPG_45_feat_0850_Red_merge = df_PPG_45_feat_0850_Red_merge[col_list]

dict = {'Sex':{'Male':1, 'Female':0}}      # label = column name
df_PPG_45_feat_0850_Red_merge.replace(dict,inplace = True) 

#################### --- Create csv : 940
df_PPG_45_feat_0850_Red_merge.to_csv('./LED-0940_7_lbs.csv')

########################################## 1070 LED ###########################


""" 
Preprocess the dataset for LED-1070_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

#df = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)
#df.to_csv('CSV_Files/LED-0850_PPG_CSV/PPG1_45_feat_0850_Red.csv')      

''' Merge PPG1_45_feat_0850_Red and Clinical data '''
import numpy as np
import pandas as pd
df_Clinical = pd.read_csv("./Shared AmaderGram Data 2018-DataSheet - Clinical data.csv")

df_PPG_45_feat_0850_Red = pd.read_csv('/media/rezwan/Study/undergraduate-thesis-non-invasive-hemoglobin-measurement/CSV_Files/LED-1070_PPG_CSV/Final_dataset/PPG1_45_feat_1070_Red_merge.csv')

df_PPG_45_feat_0850_Red_merge = pd.merge(df_PPG_45_feat_0850_Red, df_Clinical, on='ID', how='outer')

col_list = ['ID', 'Age', 'Sex', 'Systolic_peak(x)', 'Diastolic_peak(y)',
       'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x',
       'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)',
       'Dicrotic_notch_time(t3)',
       'Time_between_systolic_and_diastolic_peaks(∆T)',
       'Time_between_half_systolic_peak_points(w)',
       'Inflection_point_area_ratio(A2/A1)',
       'Systolic_peak_rising_slope(t1/x)',
       'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi',
       't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2',
       '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi',
       'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi',
       '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)',
       'Fundamental_component_magnitude(|sbase|)',
       '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)',
       '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)',
       'Stress-induced_vascular_response_index(sVRI)',  
       'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM']
print(len(col_list))

df_PPG_45_feat_0850_Red_merge = df_PPG_45_feat_0850_Red_merge[col_list]

dict = {'Sex':{'Male':1, 'Female':0}}      # label = column name
df_PPG_45_feat_0850_Red_merge.replace(dict,inplace = True) 

#################### --- Create csv : 940
df_PPG_45_feat_0850_Red_merge.to_csv('./LED-1070_7_lbs.csv')



############################################# LED - 0940 ################################

""" 
Preprocess the dataset for LED-0940_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

#df = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)
#df.to_csv('CSV_Files/LED-0850_PPG_CSV/PPG1_45_feat_0850_Red.csv')      

''' Merge PPG1_45_feat_0850_Red and Clinical data '''
import numpy as np
import pandas as pd
df_Clinical = pd.read_csv("./Shared AmaderGram Data 2018-DataSheet - Clinical data.csv")

df_PPG_45_feat_0850_Red = pd.read_csv('/media/rezwan/Study/undergraduate-thesis-non-invasive-hemoglobin-measurement/CSV_Files/LED-0940_PPG_CSV/Final_dataset/PPG1_45_feat_0940_Red_merge.csv')

df_PPG_45_feat_0850_Red_merge = pd.merge(df_PPG_45_feat_0850_Red, df_Clinical, on='ID', how='outer')

col_list = ['ID', 'Age', 'Sex', 'Systolic_peak(x)', 'Diastolic_peak(y)',
       'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x',
       'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)',
       'Dicrotic_notch_time(t3)',
       'Time_between_systolic_and_diastolic_peaks(∆T)',
       'Time_between_half_systolic_peak_points(w)',
       'Inflection_point_area_ratio(A2/A1)',
       'Systolic_peak_rising_slope(t1/x)',
       'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi',
       't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2',
       '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi',
       'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi',
       '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)',
       'Fundamental_component_magnitude(|sbase|)',
       '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)',
       '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)',
       'Stress-induced_vascular_response_index(sVRI)',  
       'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM']
print(len(col_list))

df_PPG_45_feat_0850_Red_merge = df_PPG_45_feat_0850_Red_merge[col_list]

dict = {'Sex':{'Male':1, 'Female':0}}      # label = column name
df_PPG_45_feat_0850_Red_merge.replace(dict,inplace = True) 

#################### --- Create csv : 940
df_PPG_45_feat_0850_Red_merge.to_csv('./LED-0940_7_lbs.csv')

########################################## NO LED ###########################


""" 
Preprocess the dataset for LED-1070_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

#df = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)
#df.to_csv('CSV_Files/LED-0850_PPG_CSV/PPG1_45_feat_0850_Red.csv')      

''' Merge PPG1_45_feat_0850_Red and Clinical data '''
import numpy as np
import pandas as pd
df_Clinical = pd.read_csv("./Shared AmaderGram Data 2018-DataSheet - Clinical data.csv")

df_PPG_45_feat_0850_Red = pd.read_csv('/media/rezwan/Study/undergraduate-thesis-non-invasive-hemoglobin-measurement/CSV_Files/LED-NO_PPG_CSV/Final_dataset/PPG1_45_feat_NO_Red_merge.csv')

df_PPG_45_feat_0850_Red_merge = pd.merge(df_PPG_45_feat_0850_Red, df_Clinical, on='ID', how='outer')

col_list = ['ID', 'Age', 'Sex', 'Systolic_peak(x)', 'Diastolic_peak(y)',
       'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x',
       'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)',
       'Dicrotic_notch_time(t3)',
       'Time_between_systolic_and_diastolic_peaks(∆T)',
       'Time_between_half_systolic_peak_points(w)',
       'Inflection_point_area_ratio(A2/A1)',
       'Systolic_peak_rising_slope(t1/x)',
       'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi',
       't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2',
       '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi',
       'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi',
       '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)',
       'Fundamental_component_magnitude(|sbase|)',
       '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)',
       '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)',
       'Stress-induced_vascular_response_index(sVRI)',  
       'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM']
print(len(col_list))

df_PPG_45_feat_0850_Red_merge = df_PPG_45_feat_0850_Red_merge[col_list]

dict = {'Sex':{'Male':1, 'Female':0}}      # label = column name
df_PPG_45_feat_0850_Red_merge.replace(dict,inplace = True) 

#################### --- Create csv : 940
df_PPG_45_feat_0850_Red_merge.to_csv('./LED-NO_7_lbs.csv')