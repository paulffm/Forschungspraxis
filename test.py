import numpy as np
import numpy.linalg as la
import scipy.linalg as sla



u0 = np.array([100, 80 * np.exp(np.pi * (-2j / 3)), 60 * np.exp(np.pi * (- 4j / 3))])

Z = np.zeros([3, 3])
z = np.array([0, 0, 0])
R = 1

A = np.ones((6, 6)) * 4.5

M11 = A
# on diagonal -1
M12 = -np.eye(6)
print('M12', M12)
# diagonal 1 1 1 0 0 0
M21 = sla.block_diag(np.eye(3), Z)
print('M21', M21)

M22 = np.block([[Z, Z], [np.eye(3), -np.diag([R, R, R])]])
print('M22', M22) 

M = np.block([[M11, M12], [M21, M22]])
print('M', M, M.shape)

# 12, vektor
b = np.concatenate([z, z, u0, z])
print('b', b, b.shape)