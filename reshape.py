import numpy as np 

def reshape(dataset, row, col, N):
    dataset_reshaped = np.zeros((row * col, N))
    for i in range(N):
        dataset_reshaped[:,i] = np.reshape(dataset[i], row * col)

    return dataset_reshaped