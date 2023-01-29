# import
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants


def H_analytic(r, r1, r2, I):
    h = np.zeros(len(r))
    for i, r_i in enumerate(r):
        if r_i <= r1:
            h[i] = (I / (2 * np.pi * r1 ** 2)) * r_i
        elif r_i > r1 and r_i <= r2:
            h[i] = I / (2 * np.pi * r_i)
        else:
            h[i] = 0
    return h


def B_analytic(r, r1, r2, I, mu1, mu2):
    h = np.empty(len(r))
    for i, r_i in enumerate(r):
        if r_i <= r1:
            h[i] = (I * mu1 / (2 * np.pi * r1 ** 2)) * r_i
        elif r_i > r1 and r_i <= r2:
            h[i] = I * mu2 / (2 * np.pi * r_i)
        else:
            h[i] = 0
    return h


def A_z(r, r1, r2, I, mu1, mu2):
    """
    Analytic solution of magnetic vector potential

    Parameters
    ----------
    r : np.ndarray
        radius in [m]

    Returns
    -------
    a_z : np.ndarray
        Magnetic vector potential in [Tm]
    """

    def A_z_i(r):
        return -I / (2 * np.pi) * (mu1 / 2 * (r ** 2 - r1 ** 2) / r1 ** 2 + mu2 * np.log(r1 / r2))

    def A_z_a(r):
        return -mu2 * I / (2 * np.pi) * np.log(r / r2)

    condition = r < r1
    return condition * A_z_i(r) + (~condition) * A_z_a(r)

def main():
    r1 = 2e-3
    r2 = 3.5e-3
    I = 16
    l = 300e-3
    mu1 = constants.mu_0
    mu2 = constants.mu_0 * 5
    r = np.linspace(0, r2*2, 100)
    h = H_analytic(r, r1, r2, I)
    B = B_analytic(r, r1, r2, I, mu1, mu2)
    A = A_z(r, r1, r2, I, mu1, mu2)

    L1 = mu1 * l / (8 * np.pi)
    L2 = (mu2 * l / (2 * np.pi)) * np.log(r2 / r1)
    W = 0.5 * L1 * (I ** 2) + 0.5 * L2 * (I ** 2)
    print(f'Energy: {W} Joule')

    plt.figure(3)
    plt.plot(r, A)
    plt.xlabel('r')
    plt.ylabel('A')
    plt.show()

    plt.figure(1)
    plt.plot(r, h)
    plt.xlabel('r')
    plt.ylabel('field strength H')
    plt.show()

    plt.figure(2)
    plt.plot(r, B)
    plt.xlabel('r')
    plt.ylabel('field strength B')
    plt.show()

if __name__ == "__main__":
    main()