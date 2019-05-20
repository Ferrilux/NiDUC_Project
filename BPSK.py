import cv2
import numpy as nm

image = cv2.imread('example.jpg',0) #wczytanie pliku jpg

ret, bw_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY) #konwersja na tablice binara
cv2.imshow("Binary Image",bw_img) #testowe wyswietlenie przekonwertowanego obrazu

data = bw_img #zapis danych do nowej zmiennej

print(len(data)) #wyswietlenie dlugosci tablicy (kontrolnie, mozna wywalic)
