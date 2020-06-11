from os import listdir
import cv2
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt


def pca(dataset): 

    media = np.mean(dataset, axis=1)
    Xc = dataset - media[:,np.newaxis]

    # 3b. Calcolo la matrice di covarianza 
    C = np.cov(Xc,rowvar=False)
    # 3c. Estraggo autovettori (eigenfaces) e autovalori della matrice di covarianza
    lambdas,U = np.linalg.eigh(C)
    U = Xc.dot(U)
    U = U/np.linalg.norm(U)

    # 3d. Ordino gli autovalori dal più grande al più piccolo
    best_eig_idxs = np.argsort(lambdas)[::-1]
    best_eig = lambdas[best_eig_idxs]
    best_U = U[:,best_eig_idxs]

    # 3e. Verifico la quantità di varianza dei dati che ogni autovalore porta con se e imposto N pari al numero di autovettori sufficente per avere almeno l'80% della varianza totale.
    d = lambdas.shape[0]

    y = np.cumsum(best_eig)/np.sum(best_eig)

    N = np.where(y >= 0.90)[0][0]
    print("N: ", N)
    # 3f. Proietto i dati utilizzando gli N autovettori più grandi
    Tr = best_U[:,:N]
    XT = Tr.T.dot(Xc)
    print(XT.shape)

    return Tr, XT, dataset, media