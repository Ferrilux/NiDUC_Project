#Szum.py

import numpy as nm
import matplotlib.pyplot as plt

def error_calc(lengthdata, noise_amp):
    b = nm.random.uniform(-1, 1, lengthdata)

    signal = nm.zeros(lengthdata,float)
    for i in range(len(b)):
        if b[i] < 0:
            signal[i] = -1
        else:
            signal[i]=1
    noise = nm.random.randn(lengthdata)

    rec_signal =signal + noise_amp *noise

    detected_signal = nm.zeros(lengthdata,float)
    for i in range(len(b)):
        if rec_signal[i] < 0:
          detected_signal[i]= -1
        else:
            detected_signal[i]=1

    error_matrix = abs((detected_signal - signal)/2)
    error = error_matrix.sum()
    return error