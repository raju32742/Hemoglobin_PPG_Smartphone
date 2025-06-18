#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:11:01 2018

@author: rezwan
"""

def max_first_three(lst):
    ranks = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )
    values = []
    posns = []
    for x,i in ranks:
        if x not in values:
            values.append( x )
            posns.append( i )
            if len(values) == 3:
                break
            
    return values, posns