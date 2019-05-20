import cv2
import numpy as nm
from scipy.special import erfc
import matplotlib.pyplot as plt

from Szum import error_calc

image = cv2.imread('example.jpg',0) #wczytanie pliku jpg

ret, bw_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY) #konwersja na tablice binara
cv2.imshow("Binary Image",bw_img) #testowe wyswietlenie przekonwertowanego obrazu

data = bw_img #zapis danych do nowej zmiennej
lengthdata = len(data)
print(len(data)) #wyswietlenie dlugosci tablicy (kontrolnie, mozna wywalic)
print(data)
iter_len=30 #iteracja dla lepszej średniej wartości
SNR_db = nm.array(data,float)
noise = nm.zeros(len(SNR_db),float)
error = nm.zeros((iter_len, len(SNR_db)),float)
for iter in range(iter_len):
 #Generuje różna poziomy szumów
    for i in range (len(noise)):
        noise[i]= 1/nm.sqrt(2)*10**(-SNR_db[i]/20)
     #obliczanie błędu
    error_matrix =nm.zeros(len(SNR_db),float)
    for i in range (len(noise)):
        error_matrix[i] =error_calc(lengthdata, noise[i])
     #zapisywanie błędów co interacje
    error[iter]=error_matrix
 #średni błąd
BER = error.sum(axis=0)/(iter_len*lengthdata)

theoryBER = nm.zeros(len(SNR_db),float)
for i in range (len(SNR_db)):
    theoryBER[i] = 0,5*erfc(nm.sqrt(10**(SNR_db[i]/10)))
plt.semilogy(SNR_db, BER, '--')
plt.semilogy(SNR_db, theoryBER, 'mo')
plt.ylabel('BER')
plt.xlabel('SNR')
plt.title('BPSK BER Curves')
plt.legend(['Simulation', 'Theory'], loc='upper right')

plt.grid()
plt.show()