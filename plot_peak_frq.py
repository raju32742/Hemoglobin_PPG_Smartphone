#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:18:44 2018

@author: rezwan
"""
from libraries import *

def plot_peak_frq(y, x, xm, xn, ym, yn, color = "r", label="Channel"):
    plt.figure(figsize=(8,6))
    plt.tight_layout()
    plot = pylab.plot(x,y, color, linewidth=1, label=label)
    pylab.hold(True)
    pylab.plot(xm, ym, 'k+')
    pylab.plot(xn, yn, 'm+')