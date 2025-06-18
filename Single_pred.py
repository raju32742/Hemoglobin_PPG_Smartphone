#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:59:41 2019

@author: rezwan
"""
## import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
from measrmnt_indices import *
from utils_plots import *
from utils import *
from sklearn import metrics

from libraries import *
from plot_time_series import *
from bandPass import *
from peakdetection import peakdet
from feature_PPG_N_Back_Clip import *
from max_first_three import *
from params import *
from feature import extract_ppg45, extract_svri
from video_to_frames import *
#### Seed
import random
seed = 42
np.random.seed(seed)



############################ Extract Frames.

#### Enter the Sourse and Destination folder
src_dir = "/media/rezwan/Softwares/THESIS/Extract-Images/Single_pred"
dst_dir = "/media/rezwan/Softwares/THESIS/Extract-Images/Single_pred/Frames"
#src_dir = "/media/rezwan/Softwares/THESIS/LED-Broad"
#dst_dir = "/media/rezwan/Softwares/THESIS/Extract-Images"
dataFrameArr = []

# Enter the LED-Broad
#LED_Broad_num = ["/LED-0850", "/LED-0940", "/LED-1070","/LED-NO"]
LED_Broad_num = ["/LED-0850"]

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
        print("Head: ", head)
        print("Tail: ", tail)
        # get frames from video
        path = video_to_frames(vDirName, destinationDir, tail)
        dataFrameArr.append(path)

print(dataFrameArr) ### Load the directory of all image's frame file.
################################
#     End Extract Frames       #
################################


########################===========================================

######## Feature Extrct
""" Generate .csv files for  R, G, B channels respectively . """

##############################
#   Select the directory    #
##############################
#src_dir = "Extract-Images"  
#src_dir = "RawData/Extract-Images"
src_dir = dst_dir


                    ######################################
                    #          /LED-0850                #
                    #####################################

LED_Broad_num = ["/LED-0850"]

#LED_Broad_num = ["/LED-0850", "/LED-0940", "/LED-1070","/LED-NO"] #, "/LED-0940", "/LED-1070","/LED-NO"

PPG_45_feat_0850_Red = []

for board in LED_Broad_num:
    path = src_dir + board + "/*" 
    folList = glob.glob(path)
#    print(folList)
    print("Path : " + str(path))
    
    
    idx = 0 ##################
    
    for folder in folList:
        print("Folder : " + str(folder))
        file_names = []
        for file in os.listdir(folder):
            file_names.append(file)
            
        file_names = sorted(file_names,  key = lambda x: int(x[:-4]))
        #Files.append(file_names)
#        print(file_names)
        print("###########################################")
        print("Frame numbers: " + str(len(file_names)))
#        print(file_names)
        
        
        r_mean = []
        g_mean = []
        b_mean = []
        
        seq_num = []
        
        
        file_cnt = 0
        
        len_file = len(file_names)
        
        if len_file > 600:
            for file in file_names:
                img = cv2.imread(os.path.join(folder,file))
                
                if file_cnt == 600: # Take 600 frames.
                    break
                    
                file_cnt += 1
                
                seq_num.append(int(file[:-4]))
                
                # print("Image size: " + str(img.shape))
                
                '''
                 #to get a square centered in the middle
                h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                w= img.shape[1]//2 - 250 # width//2 - 250. where width = 1920 px
                img = img[h:h+500, w:w+500]
                # img = img[290:790, 710:1210]
                '''
                
                #to get a square right to left
                h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                w= img.shape[1] # width. where width = 1920 px
                img = img[h:h+500, w-500:w]
                
                # print("Image size: " + str(img.shape))
                
                #print(file[:-4])
                
                average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
                
                #BGR
                b_mean.append(average_color[0])
                g_mean.append(average_color[1])
                r_mean.append(average_color[2])
                
        else:
            for file in file_names:
                img = cv2.imread(os.path.join(folder,file))
                
                if file_cnt == len_file - 1: # Take < 600 frames.
                    break
                    
                file_cnt += 1
                
                seq_num.append(int(file[:-4]))
                
                # print("Image size: " + str(img.shape))
                
                '''
                 #to get a square centered in the middle
                h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                w= img.shape[1]//2 - 250 # width//2 - 250. where width = 1920 px
                img = img[h:h+500, w:w+500]
                # img = img[290:790, 710:1210]
                '''
                
                #to get a square right to left
                h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                w= img.shape[1] # width. where width = 1920 px
                img = img[h:h+500, w-500:w]
                
                # print("Image size: " + str(img.shape))
                
                #print(file[:-4])
                
                average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
                
                #BGR
                b_mean.append(average_color[0])
                g_mean.append(average_color[1])
                r_mean.append(average_color[2])
                
                
        ##############################
        LED_Broad = folder.split('/')[-2].split('-')[1]
        print("LED Broad: " + str(LED_Broad))
        
        ID = folder.split('/')[-1].split('-')[0]
        print("ID: " + str(ID))
        #############################
        print(max(r_mean), min(r_mean))
        print(max(g_mean), min(g_mean))
        print(max(b_mean), min(b_mean))
        ##================ Plot all channel
        plot_all_chan(r_mean, g_mean, b_mean)

        ##################################################
        #           R channel
        ##################################################
        #plot_time_series(r_mean, "r", "R channel")
        R_bandPass = bandPass(r_mean)
        R_bandPass = R_bandPass[::-1]
        #plot_time_series(R_bandPass, "r", "BandPass For R channel")
        
        '''Peak detection from PPG'''
        from matplotlib.pyplot import plot, scatter, show
        from numpy import array
        series = R_bandPass # [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
        maxtab, mintab = peakdet(series,0.1)
        
                
        pos_maxtab = array(maxtab)[:,0]
        val_maxtab = array(maxtab)[:,1]
        pos_mintab = array(mintab)[:,0]
        val_mintab = array(mintab)[:,1]
        
        plt.style.use('ggplot') # nicer plots
        np.random.seed(52102) 
        plt.figure(figsize=(22,14))
        plot(series)

        scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
        scatter(array(mintab)[:,0], array(mintab)[:,1], color='green')
        for i, txt in enumerate(pos_maxtab):
            plt.annotate(txt, (pos_maxtab[i], val_maxtab[i]))
            
        for i, txt in enumerate(pos_mintab):
            plt.annotate(txt, (pos_mintab[i], val_mintab[i]))
        
#        plt.savefig("/media/rezwan/Softwares/THESIS/PPG_Img/LED-0850/Red_img/"+str(ID)+"_"+str(LED_Broad)+"_red.png", dpi = 100)
        plt.show()
        
        
        ''' Pick Fresh 3 PPG wave '''
        
        values, posns = max_first_three(val_maxtab)
#        print(values, posns)
        
        rez = []
        
        for idx in posns:
            left_idx = idx - 1
            right_idx = idx #+ 1
            if left_idx < 0:
                left_idx = left_idx + 1
            elif right_idx >= len(pos_mintab):
                right_idx = right_idx - 1 
                
            #print(pos_mintab[left_idx], pos_mintab[right_idx])
            l_idx = int(pos_mintab[left_idx])
            r_idx = int(pos_mintab[right_idx]) + 1
#            print(series[l_idx:r_idx])
            rez.append(list(series[l_idx:r_idx]))
            
#        print(list(rez))
#        ppg_lst = sum(rez, []) 
        Red_PPG_Clean = sum(rez, [])  
#        print(ppg_lst)
#        plot_time_series(ppg_lst, "r", "Filter PPG")
        
        plot_time_series_n_save_0850(Red_PPG_Clean, 'k', "Cleaned PPG Wave", ID, LED_Broad)
        idx += 1
        
        """ PPG-45 + extract_svri Feature Extraction """
        ppg_feat_45 = extract_ppg45(Red_PPG_Clean)
        svri = extract_svri(Red_PPG_Clean)
        
        ppg_feat_45.insert(0, int(ID))
        ppg_feat_45.append(svri)
        PPG_45_feat_0850_Red.append(ppg_feat_45)
        
       


""" 
Preprocess the dataset for LED-0850_red
"""
headers = ['ID', 'Systolic_peak(x)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)', 'Relative_augmentation_index((x-y)/x)', 'z/x', '(y-z)/x', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'Time_between_systolic_and_diastolic_peaks(∆T)', 'Time_between_half_systolic_peak_points(w)', 'Inflection_point_area_ratio(A2/A1)', 'Systolic_peak_rising_slope(t1/x)', 'Diastolic_peak_falling_slope(y/(tpi-t3))', 't1/tpi', 't2/tpi', 't3/tpi', '∆T/tpi', 'ta1', 'tb1', 'te1', 'tf1', 'b2/a2', 'e2/a2', '(b2+e2)/a2', 'ta2', 'tb2', 'ta1/tpi', 'tb1/tpi', 'te1/tpi', 'tf1/tpi', 'ta2/tpi', 'tb2/tpi', '(ta1+ta2)/tpi', '(tb1+tb2)/tpi', '(te1+t2)/tpi', '(tf1+t3)/tpi', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']       

df_in = pd.DataFrame(PPG_45_feat_0850_Red, columns=headers)

########==================== Here add your age, sex==================================================
df_in.insert(1, 'Age', float(27))
df_in.insert(2, 'Sex', 1)
#######==============================================================================================

df_in.to_csv('/media/rezwan/Softwares/THESIS/Extract-Images/Single_pred/PPG1_45_feat_0850_Red.csv')   

##########=============================

################## Test for Single prediction
df_in = pd.read_csv('/media/rezwan/Softwares/THESIS/Extract-Images/Single_pred/PPG1_45_feat_0850_Red.csv')  

#####=============Standard scaler
Xorg_in = df_in.as_matrix()  # Take one dataset: hm

#scaler_in = StandardScaler()
#Xscaled_in = scaler_in.fit_transform(Xorg_in)
### store these off for predictions with unseen data
#Xmeans_in = scaler_in.mean_
#Xstds_in = scaler_in.scale_

X_in = Xorg_in[:, 2:]
#X_in = Xscaled_in[:, 2:]
#X = np.delete(Xscaled_in, 48, axis=1)
print(X_in.shape)

################## ====================================

############ Read CSV file
dataFr = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Final_dataset/PPG1_45_feat_0850_Red_merge.csv")
dataFr.head()
dataFr.shape
dataFr.drop(dataFr.columns[[0,1]], axis=1, inplace=True)
dataFr.head()
dataFr.shape
X_in.shape
df = dataFr
#### Plot correlation matrix
from utils_plots import *
#plot_corr(df)

print(df.isnull().any().any())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().sum())

####=======================================Execute above For DNN ============================

#####=============Standard scaler
Xorg = df.as_matrix()  # Take one dataset: hm

#scaler = StandardScaler()
#Xscaled = scaler.fit_transform(Xorg)
### store these off for predictions with unseen data
#Xmeans = scaler.mean_
#Xstds = scaler.scale_

y = Xorg[:, 48]
X = np.delete(Xorg, 48, axis=1)

#y = Xscaled[:, 48]
#X = np.delete(Xscaled, 48, axis=1)

print(X.shape)
print(X_in.shape)

#####$$$$$$$$$ Apply GA and train Model
###
### import GA's files for Wrapper Analysis
from feature_selection_ga_wrap import *
from feature_selection_ga_filter import *
from fitness_function import *

### Now run for Filter analisis

#fsga = Feature_Selection_GA_Wrap(X,y, model)
fsga = Feature_Selection_GA_Filter(X,y)
pop = fsga.generate(20,50) ## population size and Generation = 10,50 (200, 100)
pp = fsga.plot_feature_set_score(50) ## Generation

print("Best Indices: " +str(pop))

#######################plot slescted features
columns_names = df.columns.values
selected_columns_names = []
get_best_ind = []
for i in range(len(pop)):
    if pop[i] == 1:
        get_best_ind.append(i)
print(len(get_best_ind))
print(get_best_ind)    
##[2, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 19, 22, 24, 27, 29, 31, 32, 34, 36, 38, 40, 45]  
##[2, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 19, 22, 24, 27, 29, 31, 32, 34, 36, 38, 40, 45]  
selected_columns_names =columns_names[get_best_ind] ##### Select the feature

values_of_selected_col = []
for val in selected_columns_names:
    values_of_selected_col.append(max(df[val]))
    
###############
#plt.rcdefaults()
plt.figure(figsize=(10,8))
fig, ax = plt.subplots()
# Example data
people = selected_columns_names
y_pos = np.arange(len(people))
performance = values_of_selected_col


ax.barh(y_pos, np.array(performance), align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Value')
ax.set_title('Selected Features')
plt.show()

###########======================================= END of Feature Selectiions Methods.
#################################=== Set it on Machine learning Model
get_best_ind = []
for i in range(len(pop)):
    if pop[i] == 1:
        get_best_ind.append(i)
        
print(len(get_best_ind))

X_selct = X[:, get_best_ind]
print(X_selct.shape)

### Import model
import keras
from learn_models import *
model =  DNN(X_selct)


###==== Train the model
NUM_EPOCHS = 100
BATCH_SIZE = 32

callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                       batch_size=32, write_graph=True, write_grads=False, 
                                       write_images=True, embeddings_freq=0, 
                                       embeddings_layer_names=None, embeddings_metadata=None)]

model.fit(X_selct, y, epochs=NUM_EPOCHS, batch_size=32,
                    shuffle=True, callbacks=callbacks, validation_split=0)

##### Save the model
###==========================================
from keras.models import load_model 
model.save('DNN_model.h5') # creates a HDF5 file 'my_model.h5' 

##===============================================
##load Model
mmodel = load_model('./DNN_model.h5')

##========

X_in_select = X_in[:, get_best_ind]

pred = mmodel.predict(X_in_select) ### Prediction of Hb Level

print("Predicted Hb(gm/dL): " + str(pred))

#print("Predicted Hb(gm/dL): " + str((pred * Xstds_in) / Xmeans_in))