#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 03:51:56 2019

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
import keras
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
from measrmnt_indices import *
from utils_plots import *
from utils import *
from sklearn import metrics
from learn_models import *
#### Seed
import random
seed = 42
np.random.seed(seed)

#######=================================
##########        0850         ########
#######################################

############ Read CSV file
dataFr = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-0850_PPG_CSV/Final_dataset/PPG1_45_feat_0850_Red_merge.csv")
dataFr.head()
dataFr.shape
dataFr.drop(dataFr.columns[[0,1]], axis=1, inplace=True)
dataFr.head()
df = dataFr
#### Plot correlation matrix
from utils_plots import *
#plot_corr(df)

print(df.isnull().any().any())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().sum())

#####=============Standard scaler
Xorg = df.as_matrix()  # Take one dataset: hm

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
## store these off for predictions with unseen data
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 48]
X = np.delete(Xscaled, 48, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

########## Import Model
model = DNN(X_train)
model.summary()


############

###====== Define a checkpoint callback :
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]

###==== Train the model
NUM_EPOCHS = 100 
BATCH_SIZE = 32

callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                       batch_size=32, write_graph=True, write_grads=False, 
                                       write_images=True, embeddings_freq=0, 
                                       embeddings_layer_names=None, embeddings_metadata=None)]

# Fit the model
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32,
                    shuffle=True, callbacks=callbacks, validation_data=(X_test,y_test))

######===== predict the values
y_test_ = model.predict(X_test).flatten()
for i in range(X_test.shape[0]):
    actual = (y_test[i] * Xstds[1]) + Xmeans[1]
    y_pred = (y_test_[i] * Xstds[1]) + Xmeans[1]
    print("Expected: {:.3f}, Pred: {:.3f}".format(actual,y_pred ))
    
actual = (y_test * Xstds[1]) + Xmeans[1]
y_pred = (y_test_ * Xstds[1]) + Xmeans[1]

################# ====== plot bland_altman_plot
bland_altman_plot(actual, y_pred)

##### === Plot estimated and predicted
r = pearsonr(actual, y_pred)
act_pred_plot(actual, y_pred, r)

###====================== Performances of model
from sklearn import metrics

### Correlation coefficient : R
print("R(CC) :" +str(pearsonr(actual, y_pred)))
#### Correllation of determination
print("R^2: " + str(metrics.r2_score(actual,y_pred))) 
### MSE
print("MSE: " + str(metrics.mean_squared_error(actual, y_pred)))
### MAE
print("MAE: " + str(metrics.mean_absolute_error(actual, y_pred)))
## RMSE
print("RMSE: " +str(rmse(actual, y_pred)))

