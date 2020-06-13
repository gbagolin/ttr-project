from sklearn.svm import SVC
from upload_dataset import upload_dataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from reshape import reshape
from pca import pca

import numpy as np
import cv2


# Inizializzo i parametri
kernel = 'rbf'
max_iteration = 1
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

counts = np.zeros(4)

for e in y_train: 
    counts[e] += 1

print(counts)

counts = np.zeros(4)

for e in y_test: 
    counts[e] += 1

print(counts)

# # matrix_x_train, x_train_reshaped, _ , mean_x_train = pca(x_train_reshaped)

# # x_test_reshaped = x_test_reshaped - mean_x_train[:,np.newaxis]
# # x_test_reshaped = matrix_x_train.T.dot(x_test_reshaped)

# # print("PCA done!")

# x_train_reshaped = x_train_reshaped.T
# x_test_reshaped = x_test_reshaped.T

# models = []

# for i in range(NUM_CLASSES):
#     models.append(SVC(kernel=kernel, max_iter=max_iteration, probability=True))

# # 5. Addestro i modelli
# for i in range(NUM_CLASSES):
#     models[i].fit(x_train_reshaped, y_train==i)

# # 6. Classifico i dati del testing set e costruisco la matrice di confusione
# predicted_scores = []
# for i in range(NUM_CLASSES):
#     predicted_scores.append(models[i].predict_proba(x_test_reshaped)[:,1])

# predicted_scores = np.asarray(predicted_scores)
# predicted = np.argmax(predicted_scores,axis=0)

# cmc = np.zeros((NUM_CLASSES,NUM_CLASSES))

# for pr,y_te in zip(predicted,y_test):
#   cmc[y_te,pr] += 1.0

# # 7. Faccio il plot della matrice di confusione e calcolo accuratezza, precision e recall media rispetto alle 10 classi

# plt.imshow(cmc)
# plt.colorbar()
# plt.xlabel("Predicted")
# plt.xticks([0,1,2,3],["0","1","2","3"])
# plt.yticks([0,1,2,3], ["0","1","2","3"])
# plt.ylabel("Real")

# accuracy = np.sum(cmc.diagonal())/np.sum(cmc)

# precision = []
# recall = []
# for i in range(NUM_CLASSES):
#   precision.append(cmc[i,i]/ np.sum(cmc[:,i]))
#   recall.append(cmc[i,i]/ np.sum(cmc[:,i]))

# precision = np.asarray(precision)
# recall  = np.asarray(recall)

# precision = np.mean(precision)
# recall = np.mean(recall)

# print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + '%')
# print('Precisione media del classificatore: ' + "{0:.2f}".format(precision))
# print('Recall media del classificatore: ' + "{0:.2f}".format(recall))

# plt.show()
