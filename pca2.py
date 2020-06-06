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
    XT = np.dot(Xc.T, Tr)
    print(XT.shape)

    return Tr,XT
    

xtr1,xt1 = pca(50, 'greater')
xtr2,xt2 = pca(5, 'cazzo')

# elderly_mean = np.mean(xt1,axis = 0)
# print(elderly_mean.shape)
# # print(elderly_mean)

# young_mean = np.mean(xt2,axis = 0)
# print(young_mean.shape)
# # print(young_mean)


# image_elder = cv2.imread('train/face-a/00001A02.jpg',cv2.IMREAD_GRAYSCALE)
# row, col = image_elder.shape

# image_reshaped = np.zeros((row*col))
# image_reshaped = np.reshape(image_elder, row*col)

# image_reshaped_tmp = np.dot(image_reshaped.T,xtr1)

# distance1 = elderly_mean - image_reshaped_tmp

# image_reshaped_tmp2 = np.dot(image_reshaped.T,xtr2)
# distance2 = young_mean - image_reshaped_tmp2

# error1 = np.sum(distance1)
# error2 = np.sum(distance2)

# print(error1 > error2)


for i in range(xt1.shape[0]):
        x = np.arange(xt1.shape[1])
        y = xt1[i,:]
        plt.scatter(x,y,c = 'r')

for i in range(xt2.shape[0]):
        x = np.arange(xt2.shape[1])
        y = xt2[i,:]
        plt.scatter(x,y,c = 'b')

plt.show()



