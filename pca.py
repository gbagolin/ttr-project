import numpy as np

import matplotlib.pyplot as plt

def getFeatures(images,info): 
    """Returns n_features eigenvalues. 

    Args:
        data (arr[]): column representing the images
        n_features ([type]): number of eigenvalues to be returned. 
    """

    #3 Applico la PCA e proietto le immagini di train in uno spazio di dimensioni N

    # 3a. Calcolo la media e centro i dati
    media = np.mean(images,axis=1) 

    # Xc = images - media[:,np.newaxis]

    # # 3b. Calcolo la matrice di covarianza 
    # C = np.cov(Xc,rowvar=False)

    # # 3c. Estraggo autovettori (eigenfaces) e autovalori della matrice di covarianza
    # lambdas,U = np.linalg.eigh(C)
    # U = Xc.dot(U)
    # U = U/np.linalg.norm(U)

    # # 3d. Ordino gli autovalori dal più grande al più piccolo
    # best_eig_idxs = np.argsort(lambdas)[::-1]
    # best_eig = lambdas[best_eig_idxs]
    # best_U = U[:,best_eig_idxs]

    # # 3e. Verifico la quantità di varianza dei dati che ogni autovalore porta con se e imposto N pari al numero di autovettori sufficente per avere almeno l'80% della varianza totale.
    # d = lambdas.shape[0]

    # fig, axs = plt.subplots(2)
    # axs[0].plot(np.arange(1,d+1),best_eig)
    # axs[0].scatter(np.arange(1,d+1),best_eig)

    # y = np.cumsum(best_eig)/np.sum(best_eig)
    # axs[1].plot(np.arange(1,d+1),y)
    # axs[1].scatter(np.arange(1,d+1),y)

    # # plt.show()

    # N = np.where(y >= info)[0][0]
    # print("N: ", N)
    # # 3f. Proietto i dati utilizzando gli N autovettori più grandi
    # Tr = best_U[:,:N]
    # XT = np.dot(Xc.T, Tr)

    # return XT,Tr,media