#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:21:23 2019

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
############ Read CSV file
dataFr = pd.read_csv("/media/rezwan/Softwares/THESIS/CSV_Files/LED-NO_PPG_CSV/Final_dataset/PPG1_45_feat_NO_Red_merge.csv")
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

###
### import GA's files for Wrapper Analysis
from feature_selection_ga_wrap import *
from feature_selection_ga_filter import *
from fitness_function import *

### Now run for Wrapper analisis
##===============================Import machine learning models
from learn_models import *
#model = LinReg() #(200, 100)
#model = RbfSVR() #(200, 100)
#model = DTR() ## Prblm
#model = PLS() ## 
#model = MLPR() ## prblm
model = RFR() ## (20,50 ) 
fsga = Feature_Selection_GA_Wrap(X,y, model)
#fsga = Feature_Selection_GA_Filter(X,y)
pop = fsga.generate(20,50 ) ## population size and Generation = 10,50
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
selected_columns_names =columns_names[get_best_ind]

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
###########=======================================END of Feature Selectiions Methods.




#################################=== Set it on Machine learning Model
get_best_ind = []
for i in range(len(pop)):
    if pop[i] == 1:
        get_best_ind.append(i)
        
print(len(get_best_ind))

X_selct = X[:, get_best_ind]
print(X_selct.shape)

model = model 
############### Apply 10-fold Cross validation
n_splits = 10
cv_set = np.repeat(-1.,X_selct.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X_selct, y):
    x_train,x_test = X_selct[train_index],X_selct[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
    model.fit(x_train,y_train)
    predicted_y = model.predict(x_test)
    print("Individual R: " +str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y
 
print("Overall R: " + str(pearsonr(y,cv_set)))

### ===== For Get real values
y = (y * Xstds[48]) + Xmeans[48]
cv_set = (cv_set * Xstds[48]) + Xmeans[48]
### ===============
print("Overall R: " + str(pearsonr(y,cv_set)))
print("R^2 Score: " + str(metrics.r2_score(y, cv_set)))
print("MAE: " + str(metrics.mean_absolute_error(y, cv_set)))
print("MSE: " + str(metrics.mean_squared_error(y, cv_set)))
print("RMSE: " + str(rmse(y, cv_set)))
print("MSLE: " + str(metrics.mean_squared_log_error(y, cv_set)))
print("EVS: " + str(metrics.explained_variance_score(y, cv_set)))


################# ====== plot bland_altman_plot
bland_altman_plot(y, cv_set)

##### === Plot estimated and predicted
r = pearsonr(y, cv_set)
act_pred_plot(y, cv_set, r)



                            ############################
                            ##    For DNN Model      ###
                            ############################

#####=============Standard scaler
Xorg = df.as_matrix()  # Take one dataset: hm

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
## store these off for predictions with unseen data
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 48]
X = np.delete(Xscaled, 48, axis=1)

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
selected_columns_names =columns_names[get_best_ind]

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
NUM_EPOCHS = 20
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
cv_set = np.repeat(-1.,X_selct.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X_selct, y):
    x_train,x_test = X_selct[train_index],X_selct[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
#    model.fit(x_train,y_train)
    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32,
                    shuffle=True, callbacks=callbacks, validation_data=(x_test, y_test))
    predicted_y = model.predict(x_test).flatten()
    print("Individual R: " +str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y
 
print("Overall R: " + str(pearsonr(y,cv_set)))

### ===== For Get real values
y = (y * Xstds[48]) + Xmeans[48]
cv_set = (cv_set * Xstds[48]) + Xmeans[48]
### ===============
print("R: " + str(pearsonr(y,cv_set)))
print("R^2 Score: " + str(metrics.r2_score(y, cv_set)))
print("MAE: " + str(metrics.mean_absolute_error(y, cv_set)))
print("MSE: " + str(metrics.mean_squared_error(y, cv_set)))
print("RMSE: " + str(rmse(y, cv_set)))
print("MSLE: " + str(metrics.mean_squared_log_error(y, cv_set)))
print("EVS: " + str(metrics.explained_variance_score(y, cv_set)))


################# ====== plot bland_altman_plot
bland_altman_plot(y, cv_set)

##### === Plot estimated and predicted
r = pearsonr(y, cv_set)
act_pred_plot(y, cv_set, r)

##### Plot Actual and Prdiction values
plt.plot(y, color='b', label = 'Actual')
plt.plot(cv_set, color='r', label = 'Predicted')
plt.xlabel("Time")
plt.ylabel("Hemoglobin Concentration(gm/dL)")
plt.legend(loc='best')
plt.show()
