import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    sig_z = sigmoid(z)
    return sig_z * (1-sig_z)