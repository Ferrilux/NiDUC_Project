import cv2
import numpy as np
import scipy
import matplotlib.pylab as plt

image = cv2.imread('example.jpg', 0)  # wczytanie pliku jpg

_, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # konwersja na tablice binara
# cv2.imshow("Binary Image",bw_img) #testowe wyswietlenie przekonwertowanego obrazu

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
data_bin = np.squeeze(np.asarray(data_bin))

# non return zero encoder
non_ret_zero = np.zeros(data_len)

for i in range(0, data_len):
    non_ret_zero[i] = 2 * data_bin[i] - 1

# PN sequence Generator
temp = np.array([np.randint(0, 2), np.randint(0, 2), np.randint(0, 2), np.randint(0, 2),
                 np.randint(0, 2), np.randint(0, 2), np.randint(0, 2)])
temp_size = 2 * len(temp) - 1

pn_seq_gen = []

for i in range(temp_size):
    pn_seq_gen.append(temp[-1])
    temp = [temp[-1] ^ temp[-2], temp[0], temp[1], temp[2], temp[3],
            temp[4], temp[5]]

for i in range(0, temp_size):
    pn_seq_gen[i] = 2 * pn_seq_gen[i] - 1

# PN sequence multiplier
pn_seq_mult = []

for i in range(0, data_len):
    for j in range(1):
        for k in range(0, temp_size):
            pn_seq_mult.append(pn_seq_gen[k] * non_ret_zero[i])

final_len = len(pn_seq_mult)

# BPSK Modulator
T = 1
t = r_[0:T:0.1]
f = 1
carrier = np.sqrt(2 * (T ** -1)) * np.sin(2 * np.pi * f * t)
carrier_len = len(carrier)
plt.plot(carrier)
plt.show()

bpsk = []
for i in range(0, final_len):
    if pn_seq_mult[i] >= 0:
        bpsk.append(carrier)
    else:
        bpsk.append(-1 * carrier)

bpsk_signal = concatenate(bpsk)

plt.figure(1)
plt.plot(bpsk_signal)
plt.xlabel('Time')
plt.ylabel('BPSK Signal')
plt.show()