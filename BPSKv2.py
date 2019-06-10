from __future__ import division
import cv2
import numpy as np
import scipy
import matplotlib.pylab as plt
import random as rd

image = cv2.imread('example.jpg', 0)  # wczytanie pliku jpg

_, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # konwersja na tablice binara
#cv2.imshow("Binary Image",bw_img) #testowe wyswietlenie przekonwertowanego obrazu

data_bin = np.empty([len(bw_img), len(bw_img[0])])
# zamiana wszystkich 255 na 1
for i in range(len(bw_img)):
    for j in range(len(bw_img[i])):
        if bw_img[i][j] > 0:
            data_bin[i][j] = 1
        else:
            data_bin[i][j] = 0

print(data_bin)
data_len = len(data_bin)
print("Długość przesyłanego ciągu bitów: ")
print(data_len * len(data_bin[0]))

# próba zwężenia do macierzy 2D
data_bin  = np.array(data_bin).flatten()

unipolar = np.array(data_bin)

# zamiana 0 na -1
trans_signal = 2*unipolar - 1

bit_dur = 1
amp_scal_factor = bit_dur/2
freq = 3/bit_dur
samples = 1000
time = np.linspace(0,5,samples)

samples_per_bit = samples/unipolar.size

dd = np.repeat(unipolar, samples_per_bit)
bb = np.repeat(trans_signal, samples_per_bit)
dw = dd
bw = bb
waveform = np.sqrt(2*amp_scal_factor/bit_dur)*np.cos(2*np.pi * freq * time)
print(bw)
BPSK = bw * waveform


f, ax = plt.subplots(4,1, sharex = True, sharey = True, squeeze = True)

ax[0].plot(time,dw)
ax[1].plot(time,bw)
ax[2].plot(time,waveform)
ax[3].plot(time,BPSK, '.')
ax[0].axis([0, 5, -1.5, 1.5])
ax[0].set_xlabel('time')
plt.show()
