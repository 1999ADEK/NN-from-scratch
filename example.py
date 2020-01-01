# This example of the neural network implementation is based on the 3rd and 4th
# assignment from the Coursera ML course by Andrew Ng.
# References:
# - https://www.coursera.org/learn/machine-learning
# - https://gist.github.com/andr0idsensei/92dd7e54a029242690555e84dca93efd

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import NN

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

nn = NN.NeuralNetwork(3, [25], 1)
weights = nn.train(X, y_encoded)
pred_y = nn.predict(weights, X)
pred_y = np.argmax(pred_y, axis=1) + 1

correct = [1 if a == b else 0 for (a, b) in zip(pred_y, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))