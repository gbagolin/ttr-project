from os import listdir
import cv2
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt


def pca(age,operator): 

    image = []
    PATH_DATASET = 'train/face-a/'

    filename_images = [f for f in listdir(PATH_DATASET) if isfile(join(PATH_DATASET, f))]

    dataset = []

    #from 0 to 30 label = 0, from 31 to 59 label = 1 

    for filename in filename_images: 
        path_image = PATH_DATASET + filename
        index = filename.find('A') + 1
        if operator == 'greater': 
            if int(filename[index:index+2]) > age:
                dataset.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
        else: 
            if int(filename[index:index+2]) < age:
                dataset.append((cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)))
        
    len_dataset = len(dataset) 
    row,col = dataset[0].shape
    print(col,row)
    dataset_reshaped = np.zeros((row*col,len_dataset))
    for i in range(len_dataset):
        dataset_reshaped[:,i] = np.reshape(dataset[i], row*col)

    media = np.mean(dataset_reshaped, axis=1)
    Xc = dataset_reshaped - media[:,np.newaxis]

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

    # fig, axs = plt.subplots(2)
    # axs[0].plot(np.arange(1,d+1),best_eig)
    # axs[0].scatter(np.arange(1,d+1),best_eig)

    y = np.cumsum(best_eig)/np.sum(best_eig)

    # axs[1].plot(np.arange(1,d+1),y)
    # axs[1].scatter(np.arange(1,d+1),y)

    # plt.show()

    N = 73
    print("N: ", N)
    # 3f. Proietto i dati utilizzando gli N autovettori più grandi
    Tr = best_U[:,:N]
    XT = Tr.T.dot(Xc)
    print(XT.shape)

    return Tr, XT, dataset_reshaped, media
    

matrix_tr_elder, features_elder, dataset_reshaped_elder, mean_elder = pca(50, 'greater')
matrix_tr_young, features_young, dataset_reshaped_young, mean_young = pca(5, 'cazzo')

dataset_reshaped_young = dataset_reshaped_young - mean_elder[:,np.newaxis]
features_young = matrix_tr_elder.T.dot(dataset_reshaped_young)



for i in range(features_young.shape[0]):
        x = np.arange(features_young.shape[1])
        y = features_young[i,:]
        plt.scatter(x,y,c = 'b')

for i in range(features_elder.shape[0]):
        x = np.arange(features_elder.shape[1])
        y = features_elder[i,:]
        plt.scatter(x,y,c = 'r')

plt.show()


# len_features_young = features_young.shape[0]
# len_features_elder = features_elder.shape[0]

# # len_bucket = max(len_features_young,len_features_elder) 

# bucket_young = np.zeros((2,len_features_young))
# bucket_elder = np.zeros((2,len_features_elder))

# for i in range(len_features_young):
#     bucket_young[0,i] = np.max(features_young[i,:])
#     bucket_young[1,i] = np.max(features_young[i,:])

# for i in range(len_features_elder):
#     bucket_elder[0,i] = np.max(features_elder[i,:])
#     bucket_elder[1,i] = np.max(features_elder[i,:])


# dataset_elder = dataset_reshaped_elder[:,:20]
# dataset_test = dataset_test - mean_elder[:,np.newaxis]

# dataset_test = matrix_tr_elder.T.dot(dataset_test)






# '''

# Calcolo soglia e distanze ...

# '''
# from scipy.spatial.distance import cdist 

# theta_elder = np.max(cdist(matrix_tr_elder.T, matrix_tr_elder.T, 'euclidean'))
# theta_young = np.max(cdist(matrix_tr_young.T, matrix_tr_young.T, 'euclidean'))

# # 5. Centro i miei dati di test

# test_data = dataset_reshaped_elder[:,:40]

# x_te = test_data -  mean_elder[:,np.newaxis]
# omega_te = matrix_tr_elder.T.dot(x_te)

# # 6. Calcolo il set di distanze epsilon
# epsilon = []
# for i in range(40):
#   tmp_test = omega_te[:,i]
#   epsilon.append(np.sqrt(np.linalg.norm(tmp_test[:,np.newaxis] - features_elder, ord=2, axis=0)))
# epsilon = np.array(epsilon)

# # 7. Ricostruisco le facce e faccio un imshow dell'originale rispetto a quella ricostruita delle prime 5 immagini!

# g = matrix_tr_elder.dot(omega_te)

# # 8. Calcolo xi per la classificazione

# xi  = np.sqrt(np.linalg.norm(g-x_te,ord=2, axis=0))

# #9. In quale dei 3 casi ci troviamo per ogni faccia di test? La faccia corrispondente è della stessa persona? Fare un check delle prime 5 facce
# fig,axs = plt.subplots(5,2)
# for i in range(5):
#   if xi[i] >= theta_elder:
#     print(str(i+1) + ": Non è una faccia!")
#   elif xi[i] < theta_elder and any(epsilon[i,:]>theta_elder):
#     print(str(i+1) + ": E' una nuova faccia!")
#   elif xi[i] < theta_elder and np.min(epsilon[i,:]) < theta_elder:
#     print(str(i+1) + ": E' una faccia conosciuta! Ora la mostro!")
#     matched_indx = np.argmin(epsilon[i,:])
    







