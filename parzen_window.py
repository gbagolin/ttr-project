from sklearn.svm import SVC
from upload_dataset import upload_dataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from reshape import reshape
from pca import pca

import numpy as np
import cv2


# Inizializzo i parametri
kernel = 'poly'
max_iteration = 100
NUM_CLASSES = 4

# Inizializzo il modello di classificazione SVM

dataset = [] 
labels = []

for i in range(7): 
    tmp_dataset, tmp_labels = upload_dataset('train/faces/{0}/'.format(i), i)
    dataset.append(tmp_dataset)
    labels.append(tmp_labels)

for list in labels:
    for i in range(len(list)):
        if list[i] == 0 or list[i] == 1: 
            list[i] = 0
        elif list[i] == 2 or list[i] == 3:
            list[i] = 1
        elif list[i] == 4 or list[i] == 5: 
            list[i] = 2 
        else: 
            list[i] = 3

training_size = 800

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(7): 
    x_train += (dataset[i])[:training_size]
    y_train += (labels[i])[:training_size]
    x_test += (dataset[i])[training_size:]
    y_test += (labels[i])[training_size:]


y_train = np.array(y_train)
y_test = np.array(y_test)

row,col = 200, 200

x_train_reshaped = reshape(x_train, row, col, len(x_train))
x_test_reshaped = reshape(x_test, row, col, len(x_test))

print(x_train_reshaped.shape,x_test_reshaped.shape)
print(len(y_train),len(y_test))

# matrix_x_train, x_train_reshaped, _ , mean_x_train = pca(x_train_reshaped)

# x_test_reshaped = x_test_reshaped - mean_x_train[:,np.newaxis]
# x_test_reshaped = matrix_x_train.T.dot(x_test_reshaped)

# print("PCA done!")


# Esercizio: implementare Parzen Windows per classificare ogni dato di test, calcolando l'accuratezza
# Provare diverse funzioni kernel (gamma): rettangolo, triangolo, gaussiano, esponenziale decrescente

# 0. Definisco una funzione per calcolare il kernel gamma

def gamma(x, ktype='rect'):
  if ktype == 'rect':
        return 1 if abs(x) <= 0.5 else 0
  elif ktype == 'tri':
        return 1-abs(x) if abs(x) <= 1 else 0   # Lascio fare
  elif ktype == 'gaussian':
        return ((2*np.pi)**(-1/2)) * np.exp(-(x**2/2)) # Lascio fare
  elif ktype == 'dexp':
        return 1/2 * np.exp(-abs(x)) # Lascio fare
  else:
        raise ValueError('Kernel type not recognized. Possible options are: "rect", "tri", "gaussian", "dexp".')


# 1. Setto i parametri
h = 0.2
ktype = 'rect'

# 2. Divido i dati di train nelle classi
c1 = x_train_reshaped[:, y_train==0] 
c2 = x_train_reshaped[:, y_train==1]
c3 = x_train_reshaped[:, y_train==2]
c4 = x_train_reshaped[:, y_train==3]

# 3. Stimo le likelihood attraverso il metodo delle Parzen Windows (Slide 10)
predicted = []

# Per ogni elemento di test, calcolo la likelihood per ognuna delle 2 classi
#...appendo a predicted la classe con likelihood maggiore
for x_te in x_test[0,:]:
    lik1 = [] 
    for x_tr in c1[0,:]: 
        lik1.append( gamma( (x_te-x_tr)/h, ktype ) )   # Lascio fare  (utilizzo la funzione gamma precedentemente definita)
    l1 = 1/h * np.mean(lik1) # Lascio fare da qua in poi, equal per la classe 2
    lik2 = [] 
    for x_tr in c2[0,:]:
        lik2.append( gamma( (x_te-x_tr)/h, ktype ) )
    l2 = 1/h * np.mean(lik2)
    lik3 = [] 
    for x_tr in c3[0,:]:
        lik3.append( gamma( (x_te-x_tr)/h, ktype ) )
    l3 = 1/h * np.mean(lik3)
    lik4 = [] 
    for x_tr in c4[0,:]:
        lik4.append( gamma( (x_te-x_tr)/h, ktype ) )
    l4 = 1/h * np.mean(lik4)
    predicted.append( np.argmax([l1,l2,l3,l4])+1 ) 

# 5. Calcolo l'accuratezza
accuracy = 0
for i in range(100):
    if predicted[i] == list(y_test)[i]:
        accuracy+=1

print('Accuratezza del classificatore con kernel ' + ktype + ': ' + "{0:.2f}".format(accuracy/len(y_test)*100) + '%')