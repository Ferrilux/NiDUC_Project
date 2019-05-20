import cv2
import numpy as nm
import scipy
import matplotlib.pylab as plt

image = cv2.imread('example.jpg',0) #wczytanie pliku jpg

_, bw_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY) #konwersja na tablice binara
cv2.imshow("Binary Image",bw_img) #testowe wyswietlenie przekonwertowanego obrazu

data = bw_img #zapis danych do nowej zmiennej

# print(len(data)) #wyswietlenie dlugosci tablicy (kontrolnie, mozna wywalic)

data_len = len(data)
data_arr = nm.array(data)
bipolar = 2*data_arr-1
bit_dur = 1
amplitude_scaling = bit_dur/2
freq = 3/bit_dur
samples = 1000
time = nm.linspace(0,5,samples)

samples_per_bit = samples/data_arr.size

dd = nm.repeat(data_arr, samples_per_bit)
bb = nm.repeat(bipolar, samples_per_bit)

dw = dd
bw = bb

waveform = nm.sqrt(2*amplitude_scaling/bit_dur) * nm.cos(2*nm.pi*freq*time)

f, ax = plt.subplots(4,1, sharex=True, sharey=True, squeeze=True)
ax[0].plot(time, dw)
ax[1].plot(time, bw)
ax[2].plot(time, waveform)
ax[3].plot(time, bpsk_w, '.')
ax[0].axis([0, 5, -1.5, 1.5])
ax[0].set_xlabel('time')
plt.show()
