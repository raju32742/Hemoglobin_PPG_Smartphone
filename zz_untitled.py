#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:53:44 2019

@author: rezwan
"""

arr = ['Sex', 'Pulse_interval(tpi)', 'Augmentation_index(y/x)',
       'Systolic_peak_time(t1)']

dff = [[-0.93126615,  0.3851156 ,  0.14416965, -0.95867564],
       [-0.93126615, -0.72069701,  0.15695801, ..., -0.26],
       [-0.93126615,  0.88451484, -0.03621248, -0.30480811],

       [ 1.07380688,  0.06407323,  0.16495669,  1.58755267],
       [ 1.07380688,  0.52780109,  0.16602107,  0.49770515]]


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = arr
y_pos = np.arange(len(people))


performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()



#import pandas
#from collections import Counter
#a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
#letter_counts = Counter(a)
#df = pandas.DataFrame.from_dict(letter_counts, orient='index')
##df.plot(kind='bar')
#import numpy
#import matplotlib.pyplot as plt
#col_lables = ['interp', 'back', 'smooth_length', 'exclude', 'adj_sens', 'orient', 'extrfwhm']
#table_vals = ['1']*len(col_lables)
##axMain = plt.subplot(2,1,1)
#axTable1 = plt.subplot(2,1,2, frameon =False)
#plt.setp(axTable1, xticks=[], yticks=[]) # a way of turning off ticks
#
##axMain.plot([1,2,3])
#tab1 = axTable1.table(cellText=[table_vals], loc='upper center', colLabels=col_lables)
#tab1.scale(1.5,1.5)
#
#plt.plot(arr, dff)
#plt.show()