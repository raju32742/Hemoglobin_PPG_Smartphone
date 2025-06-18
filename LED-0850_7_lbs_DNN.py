#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:53:29 2019

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

#### Seed
import random
seed = 42
np.random.seed(seed)
from libraries import *

df = pd.read_csv("./LED-0850_7_lbs.csv")
df.shape
df.head()
df.columns
### Drop "unnamed" and "ID" cols
df.drop(df.columns[[0,1]], axis=1, inplace=True)
df.columns
df.head()
###============================================================================

print(df.isnull().any().any())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().sum())

###============================================================================
#####=============Standard scaler
Xorg = df.as_matrix()  # Take one dataset: hm

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
## store these off for predictions with unseen data
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 48:55]
X = Xscaled[:, 0:48]
###===========================================================================
###============================= Genetic Algorithm
### import GA's files for Wrapper Analysis
from feature_selection_ga_wrap import *
from feature_selection_ga_filter import *
from fitness_function import *

### Now run for Filter analisis

#fsga = Feature_Selection_GA_Wrap(X,y, model)
fsga = Feature_Selection_GA_Filter(X,y[:, 0])  ### feature selection w.r.t hemoglobin
pop = fsga.generate(50, 100) ## (50,100), (100, 200), (200, 500) population size and Generation = 10,50 (200, 100)
pp = fsga.plot_feature_set_score(100, L_nam="LED-0850_feat") ## Generation
print("Best Indices: " +str(pop))


#######################plot slescted features
columns_names = df.columns.values
selected_columns_names = []
get_best_ind = []
for i in range(len(pop)):
    if pop[i] == 1:
        get_best_ind.append(i)
        
print(len(get_best_ind))  
selected_columns_names =columns_names[get_best_ind]
print(selected_columns_names)  


##############################=====================

#################################=== Set it on Deep learning Model
get_best_ind = []
for i in range(len(pop)):
    if pop[i] == 1:
        get_best_ind.append(i)
        
print(len(get_best_ind))

X_selct = X[:, get_best_ind]
print(X_selct.shape)  

######## ===================================================

## Import model
import keras
from learn_models import *
model =  DNN(X_selct)

###==== Train the model
NUM_EPOCHS = 100 ### 100
BATCH_SIZE = 32

callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                       batch_size=32, write_graph=True, write_grads=False, 
                                       write_images=True, embeddings_freq=0, 
                                       embeddings_layer_names=None, embeddings_metadata=None)]

## Fit the model
#history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32,
#                    shuffle=True, callbacks=callbacks, validation_data=(X_test, y_test))

############### Apply 10-fold Cross validation
n_splits = 10

#cv_set = np.repeat(-1.,X_selct.shape[0])
cv_set = np.zeros((X_selct.shape[0], 7))

skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X_selct, y):
    x_train,x_test = X_selct[train_index],X_selct[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
#    model.fit(x_train,y_train)
    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32,
                    shuffle=True, callbacks=callbacks, validation_data=(x_test, y_test))
    
    predicted_y = model.predict(x_test)# .flatten()
#    print("Individual R: " +str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y
    

###################### END of Model ================================================
'''    'Hb (gm/dL)', 'Glucose (mmd/L)', 'HbA1c (%)', 'Creatinine', 'BUN', 'SPO2', 'BPM'   '''
######### check it
print("Overall R for Hemoglobin for LED-0850: " + str(pearsonr(y[:, 0],cv_set[:, 0])))

###############################################################################
### ========================== For Get real values hemoglobin =================
y_real = (y[:, 0] * Xstds[48]) + Xmeans[48]
cv_set_pred = (cv_set[:, 0] * Xstds[48]) + Xmeans[48]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for Hemoglobin of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="hemo-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="hemo-0850_act_pred")

###############################################################################
### ========================== For Get real values Glucose =================
y_real = (y[:, 1] * Xstds[49]) + Xmeans[49]
cv_set_pred = (cv_set[:, 1] * Xstds[49]) + Xmeans[49]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for Glucose of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="Glucose-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="Glucose-0850_act_pred")

###############################################################################
### ========================== For Get real values HbA1c =================
y_real = (y[:, 2] * Xstds[50]) + Xmeans[50]
cv_set_pred = (cv_set[:, 2] * Xstds[50]) + Xmeans[50]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for HbA1c of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="HbA1c-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="HbA1c-0850_act_pred")

###############################################################################
### ========================== For Get real values Creatinine =================
y_real = (y[:, 3] * Xstds[51]) + Xmeans[51]
cv_set_pred = (cv_set[:, 3] * Xstds[51]) + Xmeans[51]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for Creatinine of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="Creatinine-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="Creatinine-0850_act_pred")


###############################################################################
### ========================== For Get real values BUN =================
y_real = (y[:, 4] * Xstds[52]) + Xmeans[52]
cv_set_pred = (cv_set[:, 4] * Xstds[52]) + Xmeans[52]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for BUN of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="BUN-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="BUN-0850_act_pred")

###############################################################################
### ========================== For Get real values SPO2 =================
y_real = (y[:, 5] * Xstds[53]) + Xmeans[53]
cv_set_pred = (cv_set[:, 5] * Xstds[53]) + Xmeans[53]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for SPO2 of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="SPO2-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="SPO2-0850_act_pred")

###############################################################################
### ========================== For Get real values BPM =================
y_real = (y[:, 6] * Xstds[54]) + Xmeans[54]
cv_set_pred = (cv_set[:, 6] * Xstds[54]) + Xmeans[54]
################# ====== plot bland_altman_plot
#bland_altman_plot(y_real, cv_set_pred, "hb-0850")
 
print("Performance for BPM of LED-0850: ")
print("R: " + str(pearsonr(y_real,cv_set_pred)))
print("R^2 Score: " + str(metrics.r2_score(y_real,cv_set_pred)))
print("MAE: " + str(metrics.mean_absolute_error(y_real,cv_set_pred)))
print("MSE: " + str(metrics.mean_squared_error(y_real,cv_set_pred)))
print("RMSE: " + str(rmse(y_real,cv_set_pred)))
print("MSLE: " + str(metrics.mean_squared_log_error(y_real,cv_set_pred)))
print("EVS: " + str(metrics.explained_variance_score(y_real,cv_set_pred)))

################# ====== plot bland_altman_plot
bland_altman_plot(y_real, cv_set_pred, name="BPM-0850_bland")

##### === Plot estimated and predicted
r = pearsonr(y_real, cv_set_pred)
act_pred_plot(y_real, cv_set_pred, r, name="BPM-0850_act_pred")
 