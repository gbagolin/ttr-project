from sklearn.svm import SVC
from upload_dataset import upload_dataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import cv2

from feature_exctraction.resnet import FeaturesExtractor

# Inizializzo i parametri

kernel = 'sigmoid'
max_iteration = 1

# Inizializzo il modello di classificazione SVM

dataset, labels = upload_dataset('train/face-a/', 20, 'minor')

x_train = dataset[:,:4000]
y_train = labels[:4000]

x_test = dataset[:,4000:5000]
y_test = labels[4000:5000]


x_train = x_train.T
x_test = x_test.T
# for i in range(2):
#     image = x_train[: , i].reshape((200,200))
#     plt.imshow(image, cmap='gray')
#     plt.show()


# x_vecchi = x_train[:,y_train == 0]
# x_giovani = x_train[:,y_train == 1]

# plt.scatter(np.arange(x_vecchi.shape[0]), x_vecchi[:,0], c = 'r')
# plt.scatter(np.arange(x_giovani.shape[0]), x_giovani[:,1], c = 'b')
# plt.show()

model = SVC(kernel=kernel, max_iter=max_iteration) 
model.fit(x_train, y_train)

cmc = np.zeros((2,2))

predicted = model.predict(x_test)

for pr,y_te in zip(predicted,y_test):
  cmc[y_te,pr] += 1.0

print(cmc)

plt.imshow(cmc)
plt.colorbar()
plt.xlabel("Predicted")
plt.xticks([0,1],["6","9"])
plt.yticks([0,1],["6","9"])
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







