import numpy as np

a = np.zeros((5, 1))
b = np.ones((5, 1))
c = np.ones((5, 1)) * 3
d = np.concatenate((a, b, c), axis=1)
print(d, d.shape)