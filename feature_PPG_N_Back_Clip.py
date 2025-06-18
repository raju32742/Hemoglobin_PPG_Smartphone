#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:38:12 2018

@author: rezwan
"""

import numpy as np
from scipy.signal import argrelmax, argrelmin, welch
from params_PPG_N_Back_Clip import *



def extract_ppg45(single_waveform, sample_rate=PPG_SAMPLE_RATE):
    def __next_pow2(x):
        return 1<<(x-1).bit_length()
    features = []
    maxima_index = argrelmax(np.array(single_waveform))[0]
    minima_index = argrelmin(np.array(single_waveform))[0]
    derivative_1 = np.diff(single_waveform, n=1) * sample_rate
    derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
    derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
    derivative_2 = np.diff(single_waveform, n=2) * sample_rate
    derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
    derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]
    sp_mag = np.abs(np.fft.fft(single_waveform, n=__next_pow2(len(single_waveform))*16))
    freqs = np.fft.fftfreq(len(sp_mag))
    sp_mag_maxima_index = argrelmax(sp_mag)[0]
    # x
    x = single_waveform[maxima_index[0]]
    features.append(x)
    # y
    y = single_waveform[maxima_index[1]]
    features.append(y)
    # z
    z = single_waveform[minima_index[0]]
    features.append(z)
    # t_pi
    t_pi = len(single_waveform) / sample_rate
    features.append(t_pi)
    # y/x
    features.append(y / x)
    # (x-y)/x
    features.append((x - y) / x)
    # z/x
    features.append(z / x)
    # (y-z)/x
    features.append((y - z) / x)
    # t_1
    t_1 = (maxima_index[0] + 1) / sample_rate
    features.append(t_1)
    # t_2
    t_2 = (minima_index[0] + 1) / sample_rate
    features.append(t_2)
    # t_3
    t_3 = (maxima_index[1] + 1) / sample_rate
    features.append(t_3)
    # delta_t
    delta_t = t_3 - t_2
    features.append(delta_t)
    # width
    single_waveform_halfmax = max(single_waveform) / 2
    width = 0
    for value in single_waveform[maxima_index[0]::-1]:
        if value >= single_waveform_halfmax:
            width += 1
        else:
            break
    for value in single_waveform[maxima_index[0]+1:]:
        if value >= single_waveform_halfmax:
            width += 1
        else:
            break
    features.append(width / sample_rate)
    # A_2/A_1
    features.append(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
    # t_1/x
    features.append(t_1 / x)
    # y/(t_pi-t_3)
    features.append(y / (t_pi - t_3))
    # t_1/t_pi
    features.append(t_1 / t_pi)
    # t_2/t_pi
    features.append(t_2 / t_pi)
    # t_3/t_pi
    features.append(t_3 / t_pi)
    # delta_t/t_pi
    features.append(delta_t / t_pi)
    # t_a1
    t_a1 = derivative_1_maxima_index[0] / sample_rate
    features.append(t_a1)
    # t_b1
    t_b1 = derivative_1_minima_index[0] / sample_rate
    features.append(t_b1)
    # t_e1
    t_e1 = derivative_1_maxima_index[1] / sample_rate
    features.append(t_e1)
    # t_f1
    t_f1 = derivative_1_minima_index[1] / sample_rate
    features.append(t_f1)
    # b_2/a_2
    a_2 = derivative_2[derivative_2_maxima_index[0]]
    b_2 = derivative_2[derivative_2_minima_index[0]]
    features.append(b_2 / a_2)
    # e_2/a_2
    e_2 = derivative_2[derivative_2_maxima_index[1]]
    features.append(e_2 / a_2)
    # (b_2+e_2)/a_2
    features.append((b_2 + e_2) / a_2)
    # t_a2
    t_a2 = derivative_2_maxima_index[0] / sample_rate
    features.append(t_a2)
    # t_b2
    t_b2 = derivative_2_minima_index[0] / sample_rate
    features.append(t_b2)
    # t_a1/t_pi
    features.append(t_a1 / t_pi)
    # t_b1/t_pi
    features.append(t_b1 / t_pi)
    # t_e1/t_pi
    features.append(t_e1 / t_pi)
    # t_f1/t_pi
    features.append(t_f1 / t_pi)
    # t_a2/t_pi
    features.append(t_a2 / t_pi)
    # t_b2/t_pi
    features.append(t_b2 / t_pi)
    # (t_a1-t_a2)/t_pi
    features.append((t_a1 - t_a2) / t_pi)
    # (t_b1-t_b2)/t_pi
    features.append((t_b1 - t_b2) / t_pi)
    # (t_e1-t_2)/t_pi
    features.append((t_e1 - t_2) / t_pi)
    # (t_f1-t_3)/t_pi
    features.append((t_f1 - t_3) / t_pi)
    # f_base
    f_base = freqs[sp_mag_maxima_index[0]] * sample_rate
    features.append(f_base)
    # sp_mag_base
    sp_mag_base = sp_mag[sp_mag_maxima_index[0]] / len(single_waveform)
    features.append(sp_mag_base)
    # f_2
    f_2 = freqs[sp_mag_maxima_index[1]] * sample_rate
    features.append(f_2)
    # sp_mag_2
    sp_mag_2 = sp_mag[sp_mag_maxima_index[1]] / len(single_waveform)
    features.append(sp_mag_2)
    # f_3
    f_3 = freqs[sp_mag_maxima_index[2]] * sample_rate
    features.append(f_3)
    # sp_mag_3
    sp_mag_3 = sp_mag[sp_mag_maxima_index[2]] / len(single_waveform)
    features.append(sp_mag_3)
    return features


def extract_svri(single_waveform):
    def __scale(data):
        data_max = max(data)
        data_min = min(data)
        return [(x - data_min) / (data_max - data_min) for x in data]
    max_index = np.argmax(single_waveform)
    single_waveform_scaled = __scale(single_waveform)
    return np.mean(single_waveform_scaled[max_index:]) / np.mean(single_waveform_scaled[:max_index])


def extract_average_skin_conductance_level(signal):
    return np.mean(signal)


def extract_minimum_skin_conductance_level(signal):
    return min(signal)


def extract_average_rri(rri):
    return np.mean(rri)


def extract_rmssd(rri):
    return np.sqrt(np.mean(np.square(np.diff(rri))))


def extract_hrv_power(rri, sample_rate):
    f, psd = welch(rri, sample_rate, nperseg=len(rri))
    f_step = f[1] - f[0]
    lf_hrv_power = 0
    hf_hrv_power = 0
    total_hrv_power = 0
    for x in zip(f.tolist(), psd.tolist()):
        if x[0] >= ECG_LF_HRV_CUTOFF[0] and x[0] < ECG_LF_HRV_CUTOFF[1]:
            lf_hrv_power += x[1]
            total_hrv_power += x[1]
        elif x[0] >= ECG_HF_HRV_CUTOFF[0] and x[0] <= ECG_HF_HRV_CUTOFF[1]:
            hf_hrv_power += x[1]
            total_hrv_power += x[1]
        elif x[0] > ECG_HF_HRV_CUTOFF[1]:
            total_hrv_power += x[1]
    return lf_hrv_power / total_hrv_power * 100.0, hf_hrv_power / total_hrv_power * 100.0