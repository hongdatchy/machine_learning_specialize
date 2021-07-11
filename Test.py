import numpy as np

# x = np.random.rand(3, 2)
# # print(x)
# # init a matrix 3x2, each element is between 0 and 1
#
#
# y = np.zeros((10, 2))
# # print(y)
# # init a matrix 3x2, each element is 0
#
# print(x)
# print(np.reshape(x, (2, 3)))
# print(np.reshape(x, (2, -1)))
# # 2 line are the same
# print(np.reshape(x, (2, -1)).T)
# # .T --> transpose in matrix
# z = np.array([
#     [[1, 2, 3]], [[3, 4, 5]]
# ])
#
# print(np.squeeze(z))
# # khong hieu

# x = np.random.rand(3, 2)
# print(x)
# x = x[:, 0]
# # all row but only select column 0
# print(x)

np.random.seed(1)
x = np.random.rand(1, 2)
print(x)
np.random.seed(1)
y = np.random.rand(1, 2)
print(y)
np.random.seed(100)
z = np.random.rand(1, 2)
print(z)

# --> np.random.seed(n) : if n is the same, np.random.rand is the same too

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
