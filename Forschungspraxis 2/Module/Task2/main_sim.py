import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mTLM import decompose, Z_ch, beta_m, A_m, A, AU, Z, Y


def main():
    f = 1e3
    l = 2e3
    R = np.eye(3) * 1e-3 / l
    G = np.zeros((3, 3)) / l
    L = np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]]) * 1e-6 / l
    C = np.array([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]]) * 1e-9 / l
    Z_m = Z(R, L, f)
    Y_m = Y(G, C, f)

    Zm, Ym, Tu, Ti = decompose(Z_m, Y_m)
    z_char = Z_ch(Zm, Ym)
    b = beta_m(Zm, Ym)
    am = A_m(z_char, b, l)
    a = A(am, Tu, Ti)
    au = AU(a, 1)

    # Task 2 - Phase plot
    # i = Ti[:, 2]
    # X = [0, 0, 0]
    # re = [np.real(z) for z in i]
    # im = [np.imag(z) for z in i]
    # plt.quiver(X, X, re, im)
    # plt.show()

    # Task 3
    u0 = [100, 80, 60]
    i0 = u0 / z_char  # TODO: Not correct!
    # x0 = np.concatenate([u0, i0], axis=None)
    x0 = np.concatenate([u0[0], i0[0], u0[1], i0[1], u0[2], i0[2]], axis=None)

    x = a @ x0
    u = x[0:3]
    i = x[3:6]



if __name__ == '__main__':
    main()
