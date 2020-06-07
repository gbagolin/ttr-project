from sklearn.svm import SVC
from upload_dataset import upload_dataset
import numpy as np

# Inizializzo i parametri

kernel = 'linear'
max_iteration = 1000

# Inizializzo il modello di classificazione SVM
model = SVC(kernel=kernel, max_iter=max_iteration) 

dataset, labels = upload_dataset('train/face-a/', 40, 'minor')

x_train = dataset[:,:100]
y_train = labels[:100]

x_test = dataset[:,100:110]
y_test = labels[100:110]

print(x_train.shape)


from matplotlib import pyplot as plt


x_vecchi = x_train[:,y_train == 0]
x_giovani = x_train[:,y_train == 1]

plt.scatter(np.arange(x_vecchi.shape[0]), x_vecchi[:,0], c = 'r')
plt.scatter(np.arange(x_giovani.shape[0]), x_giovani[:,1], c = 'b')
plt.show()



