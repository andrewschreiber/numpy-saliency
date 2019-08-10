import numpy as np


def cross_entropy(inputs, labels):
    out_num = labels.shape[0]
    probability = np.sum(labels.reshape(1, out_num) * inputs, dtype=np.float64)
    loss = -np.log(probability, dtype=np.float64)
    return loss
