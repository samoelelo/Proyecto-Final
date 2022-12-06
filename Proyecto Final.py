# -*- coding: utf-8 -*-
"""
Created on Dec 2022

@author: Samuel Romero Santiago
"""

##se cargan las librerías necesarias
import numpy as np 
import nibabel as nib##librería para cargar DTI
from numpy import linalg as LA##Se carga linalg para poder hacer el cálculo de eigenvectores y valores
from math import sqrt
##se carga la imagen DTI
img = nib.load("N9_p28_tensor.nii")
##se convierte a arreglo numpy 
img1 = img.get_fdata()
##observamos las dimenssiones de la arreglo
np.shape(img1)

#creamos un arreglo de ceros con las primeras tres dimensiones que corresponde
#a las dimensiones espaciales de img1
fa=np.zeros(np.shape(img1)[0:3])
#corroboramos
np.shape(fa)

##creamos dos funciones
#esta convierte un vector de tamaño 6 en una matriz de covarianzas
def tomatrix(arr):
    mat=np.zeros((3,3))
    mat[0][0]=arr[0]
    mat[1][1]=arr[1]
    mat[2][2]=arr[2]
    mat[0][1]=mat[1][0]=arr[3]
    mat[0][2]=mat[2][0]=arr[4]
    mat[1][2]=mat[2][1]=arr[5]
    return mat

##esta calcula Fa a partir de eigenvectores
def calfa(a,b,c):
    if a==0 and b==0 and c==0:##Es necesaria esta parte para evitar divisiones entre 0
        return 0
    else:
        tra=(a+b+c)/3##promedio de traza
        fa=sqrt(3*((a-tra)**2+(b-tra)**2+(c-tra)**2)/(2*(a**2+b**2+c**2)))
        return fa

##se hacen ciclos para todos las 3 dimensiones
for i in range(0,fa.shape[0]):
    for j in range(0,fa.shape[1]):
        for k in range(0,fa.shape[2]): 
            x=img1[i][j][k]##se fija nuestro voxel
            w, v = LA.eig(tomatrix(x))##se le calculan eigenvectores usando nuestrafuncion tomatriz
            fa[i][j][k]=calfa(w[0],w[1],w[2])##se calcula FA
            
faorig= nib.load("N9_p28_fa.nii") ##se carga nuestra imagen original FA para comparar
faorig1= faorig.get_fdata()#se pasa a arreglo numpy

##se crea una variable en la que se guardará la suma
suma=0
##se hace la suma de la resta de cada voxel de FA original menos el 
#calculado y se eleva al cuadrado, se suma a través de todo el arreglo
for i in range(0,fa.shape[0]):
    for j in range(0,fa.shape[1]):
        for k in range(0,fa.shape[2]): 
            suma+=(fa[i][j][k]-faorig1[i][j][k])**2
##se imprime la suma tiene que se 0 o muy cercana
print(suma)