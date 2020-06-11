from sklearn.svm import SVC
from upload_dataset import upload_dataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from reshape import reshape
from pca import pca

import numpy as np
import cv2


# Inizializzo i parametri

NUM_CLASSES = 3

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
        else:
            list[i] = 2 

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

matrix_x_train, x_train_reshaped, _ , mean_x_train = pca(x_train_reshaped)

x_test_reshaped = x_test_reshaped - mean_x_train[:,np.newaxis]
x_test_reshaped = matrix_x_train.T.dot(x_test_reshaped)

print("PCA done!")

# x_train_reshaped = x_train_reshaped.T
# x_test_reshaped = x_test_reshaped.T

# 1. Inizializzo i parametri

print("KNN started!")

K = 70
fun = 'euclidean'

# 2. Calcolo la distanza fra tutti gli oggetti di train e gli oggetti di test
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

from scipy.spatial.distance import cdist

D = cdist(x_train_reshaped,x_test_reshaped,metric=fun)
 
print("Distance done!")

# 3. Per ogni dato di test (argomento axis=0), ordino le distanze dalla più piccola alla più grande e trovo gli indici di train dei più vicini
# Attenzione: Tengo solo i primi K !!

neighbors = np.argsort(D, axis=0)

k_neighbors = neighbors[:K, :]

# 4. Controllo le etichette di questi K punti: devo trovare la più frequente:
#     - Ottengo le etichette dei punti vicini
#     - Trovo l'etichetta più frequente! Utilizzo la moda!

neighbors_labels = y_train[k_neighbors]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

from scipy import stats
prediction = stats.mode(neighbors_labels, axis=0)[0]

# 5. Calcolo l'accuratezza
accuracy = np.sum(prediction == y_test) / len(y_test)
print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + '%')