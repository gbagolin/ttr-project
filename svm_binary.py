from sklearn.svm import SVC
from upload_dataset import upload_dataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from reshape import reshape
from pca import pca

import numpy as np
import cv2


# test = []
# test.append(cv2.resize(cv2.imread('train/8.jpg', 0), (200,200)))
# test.append(cv2.resize(cv2.imread('train/6.jpg', 0), (200,200)))

# test_reshaped = reshape(test, 200, 200, len(test))

# test_reshaped = test_reshaped.T

# test_labels = np.array([0, 1])


# Inizializzo i parametri
kernel = 'rbf'
max_iteration = 1

# Inizializzo il modello di classificazione SVM

dataset_0, labels_0 = upload_dataset('train/faces/0/', 0)

dataset_6, labels_6 = upload_dataset('train/faces/6/', 1)

training_size = 800

x_train = dataset_0[:training_size] + dataset_6[:training_size]
y_train = np.array(labels_0[:training_size] + labels_6[:training_size])

x_test = dataset_0[training_size:] + dataset_6[training_size:] 
y_test = np.array(labels_0[training_size:] + labels_6[training_size:])

row,col = 200, 200

x_train_reshaped = reshape(x_train, row, col, len(x_train))
x_test_reshaped = reshape(x_test, row, col, len(x_test))

# matrix_x_train, x_train_reshaped, _ , mean_x_train = pca(x_train_reshaped)

# x_test_reshaped = x_test_reshaped - mean_x_train[:,np.newaxis]
# x_test_reshaped = matrix_x_train.T.dot(x_test_reshaped)

# print("PCA done!")

x_train_reshaped = x_train_reshaped.T
x_test_reshaped = x_test_reshaped.T


x_vecchi = x_train_reshaped[y_train == 1 , : ]
x_giovani = x_train_reshaped[y_train == 0 , : ]

print("Train vecchi: ", x_train_reshaped[y_train == 1, :].shape[0])
print("Train giovani: ", x_train_reshaped[y_train == 0 ,:].shape[0])

print("Test, vecchi: ", x_test_reshaped[y_test == 1, :].shape[0])
print("Test giovani: ", x_test_reshaped[y_test == 0 ,:].shape[0])

model = SVC(kernel=kernel, max_iter=max_iteration) 
model.fit(x_train_reshaped, y_train)

cmc = np.zeros((2,2))

predicted = model.predict(x_test_reshaped)

for pr,y_te in zip(predicted,y_test):
  cmc[y_te,pr] += 1.0

print(cmc)

plt.imshow(cmc)
plt.colorbar()
plt.xlabel("Predicted")
plt.xticks([0,1],["Giovani","Vecchi"])
plt.yticks([0,1],["Giovani","Vecchi"])
plt.ylabel("Real")
plt.show()

accuracy = np.sum(cmc.diagonal())/np.sum(cmc)

precision_0 = cmc[0,0] / np.sum(cmc[:,0])
precision_1 = cmc[1,1]/ np.sum(cmc[:,1])

recall_0 = cmc[0,0]/ np.sum(cmc[0,:])
recall_1 = cmc[1,1]/ np.sum(cmc[1,:])

print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + '%')
print('Precisione del classificatore rispetto alla classe 0: ' + "{0:.2f}".format(precision_0))
print('Precisione del classificatore rispetto alla classe 1: ' + "{0:.2f}".format(precision_1))
print('Recall del classificatore rispetto alla classe 0: ' + "{0:.2f}".format(recall_0))
print('Recall del classificatore rispetto alla classe 1: ' + "{0:.2f}".format(recall_1))







