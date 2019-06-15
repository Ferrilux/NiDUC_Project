from __future__ import division
import cv2
import numpy as np
import scipy
import matplotlib.pylab as plt
import random as rd


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

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
dlugoscNaszegoSygnalu = len(data_bin)
print("Długość przesyłanego ciągu bitów: ")
print(data_len * len(data_bin[0]))
naszSygnal= data_bin

# próba zwężenia do macierzy 2D
data_bin  = np.array(data_bin).flatten()

unipolar = np.array(data_bin)
amplituda = 10
zakluceniaAmp = 0.25
zakluceniaFazy = 0.5
t = np.arange(0.1,len(naszSygnal),0,1)
x = np.arange(1,(dlugoscNaszegoSygnalu+2)*100,1)
for i in range (len(naszSygnal)):
    zakluceniaAmp[i] = rd.random(zakluceniaAmp);
    zakluceniaFazy[i] = rd.random(zakluceniaFazy);
zmienionyNaszSygnal = 0
pomocniczy = 0
pomocniczyParzysty = 0
pomocniczyNieparzysty = 0

for i in range(dlugoscNaszegoSygnalu):
    if(naszSygnal[i] == 0):
        zmienionyNaszSygnal[i] = -1
    else:
        zmienionyNaszSygnal[i] = 0
    for j in range(np.arange(i,i+1,0.1)):
        pomocniczy[x*np.arange(i*100,(i+1)*100,1)] = zmienionyNaszSygnal[i]
        if i % 2 ==0 :
            pomocniczyParzysty[x*np.arange(i*100,(i+1)*100,1)] = zmienionyNaszSygnal[i]
            pomocniczyParzysty[x * np.arange((i+1) * 100, (i + 2) * 100, 1)] = zmienionyNaszSygnal[i]
        else:
            pomocniczyNieparzysty[x * np.arange(i * 100, (i + 1) * 100, 1)] = zmienionyNaszSygnal[i]
            pomocniczyNieparzysty[x * np.arange((i + 1) * 100, (i + 2) * 100, 1)] = zmienionyNaszSygnal[i]
cyfrowyPomocniczyParzysty = 0
cyfrowy =0
cyfrowyPomocniczyNieparzysty = 0
for i in range(dlugoscNaszegoSygnalu):
    if (naszSygnal[i] == 0):
        zmienionyNaszSygnal[i] = -1
    else:
        zmienionyNaszSygnal[i] = 0
    for j in range(np.arange(i, i + 1, 0.1)):
        cyfrowy[x * np.arange(i * 100, (i + 1) * 100, 1)] = zmienionyNaszSygnal[i]
        if i % 2 == 0:
            cyfrowyPomocniczyParzysty[x * np.arange(i * 100, (i + 1) * 100, 1)] = zmienionyNaszSygnal[i]
            cyfrowyPomocniczyParzysty[x * np.arange((i + 1) * 100, (i + 2) * 100, 1)] = zmienionyNaszSygnal[i]
        else:
            cyfrowyPomocniczyNieparzysty[x * np.arange(i * 100, (i + 1) * 100, 1)] = zmienionyNaszSygnal[i]
            cyfrowyPomocniczyNieparzysty[x * np.arange((i + 1) * 100, (i + 2) * 100, 1)] = zmienionyNaszSygnal[i]
pomocniczy = pomocniczy[np.arange(101,len(pomocniczy),1)]
pomocniczyParzysty = pomocniczyParzysty(np.arange(201,(dlugoscNaszegoSygnalu+2)*100,1))
pomocniczyNieparzysty = pomocniczyNieparzysty(np.arange(101,(dlugoscNaszegoSygnalu+1)*100,1))

cyfrowy = cyfrowy[np.arange(101,len(pomocniczy),1)]
cyfrowyPomocniczyParzysty = cyfrowyPomocniczyParzysty(np.arange(201,(dlugoscNaszegoSygnalu+2)*100,1))
cyfrowyPomocniczyNieparzysty = cyfrowyPomocniczyNieparzysty(np.arange(101,(dlugoscNaszegoSygnalu+1)*100,1))
parzyste=0
nieparzyste=0
pParzyste=0
pNieparzyste =0;
cpParzyste = 0
cpNieparzyste =0
for i in range((len(pomocniczyParzysty)/2)):
    parzyste[i] = 2*i
    nieparzyste[i] = 2*i -1
    pParzyste[i]=pomocniczyParzysty[parzyste[i]]
    pNieparzyste[i]=pomocniczyNieparzysty[nieparzyste[i]]
    cpParzyste[i]=cyfrowyPomocniczyParzysty[parzyste[i]]
    cpNieparzyste[i]=cyfrowyPomocniczyNieparzysty[nieparzyste[i]]

x =1
y =100
ct = 0
st = 0
cost =0
sint = 0
for i in range(dlugoscNaszegoSygnalu):
    for j in range(y):
        cost[j] = np.cos(2*np.pi*t[j]+zakluceniaFazy[i])
        sint[j] = np.sin(2*np.pi*t[j]+zakluceniaFazy[i])
    x=x+100
    y=y+100
x=1
y=100
for i in range(dlugoscNaszegoSygnalu):
    for j in range(y):
        ct[j] = pomocniczy[j]*cost[j]*(amplituda+zakluceniaAmp[i])
        st[j] = pomocniczy[j]*sint[j]*(amplituda+zakluceniaAmp[i])
    x=x+100
    y=y+100
y=100
ctP=0
ctN=0
stP=0
stN=0
for i in range((dlugoscNaszegoSygnalu/2)):
    for j in range(y):
        ctP[j] = pParzyste[j]*cost[j]*(amplituda+zakluceniaAmp[i])
        ctN[j] = pNieparzyste[j] * cost[j] * (amplituda + zakluceniaAmp[i])
        ctP[j] = pParzyste[j] * sint[j] * (amplituda + zakluceniaAmp[i])
        ctP[j] = pNieparzyste[j] * sint[j] * (amplituda + zakluceniaAmp[i])
    y=y+100

qpsk = ctP+stN
punktyXbpsk = 0
punktyYbpsk = 0
j=1
for i in range(dlugoscNaszegoSygnalu) tb cnd






