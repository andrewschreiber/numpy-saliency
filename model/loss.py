import numpy as np


def cross_entropy(inputs, labels):
    out_num = labels.shape[0]
    probability = np.sum(labels.reshape(1, out_num) * inputs)
    loss = -np.log(probability)
    return loss
