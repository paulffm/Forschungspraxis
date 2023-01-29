import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from numpy import pi, sqrt


def Z(R: np.ndarray, L: np.ndarray, f: float) -> np.ndarray:
    """The impedance Z."""
    return R + 2j * pi * f * L


def Y(G: np.ndarray, C: np.ndarray, f: float) -> np.ndarray:
    """The admittance Y."""
    return G + 2j * pi * f * C


def decompose(Z: np.ndarray, Y: np.ndarray) -> [np.ndarray, np.ndarray]:
    """Calculates the modal decomposition of Z and Y."""
    u, Tu = la.eig(Z @ Y)
    i, Ti = la.eig(Y @ Z)

    Zm = la.inv(Tu) @ Z @ Ti
    Ym = la.inv(Ti) @ Y @ Tu
    return Zm, Ym, Tu, Ti


def Z_ch(Zm: np.ndarray, Ym: np.ndarray) -> np.ndarray:
    """The characteristic impedance values for each cable for the given modal Z and Y."""
    return np.diag(sqrt(Zm / Ym))


def beta_m(Zm: np.ndarray, Ym: np.ndarray) -> np.ndarray:
    """Calculates the propagation constants for each cable gamma for the given modal Z and Y."""
    return np.diag(sqrt(Zm * Ym))


def Ak_m(Zkm_ch: float, bkm: float, l: float) -> np.ndarray:
    """Calculates the local modal propagation matrix for the given line parameters."""
    bl = bkm * l
    return np.array([
        [np.cosh(bl), -Zkm_ch * np.sinh(bl)],
        [-np.sinh(bl) / Zkm_ch, np.cosh(bl)]
    ])


def A_m(Zm_ch: np.ndarray, bm: np.ndarray, l: float) -> np.ndarray:
    """Calculates the modal propagation matrix.
    :param Zm_ch: Array of characteristic impedance values.
    :param bm: Array of propagation constants.
    :param l: Length of the cable.
    """
    ak_m = [Ak_m(Zkm, bkm, l) for Zkm, bkm in zip(Zm_ch, bm)]
    return sla.block_diag(ak_m[0], ak_m[1], ak_m[2])


def A(A_m: np.ndarray, Tu: np.ndarray, Ti: np.ndarray) -> np.ndarray:
    """Calculates the propagation matrix."""
    zero = np.zeros((3, 3))
    T = np.block([
        [Tu, zero],
        [zero, Ti]
    ])
    return T @ A_m @ la.inv(T)


def AU(A: np.ndarray, R: float) -> np.ndarray:
    """Calculates the propagation matrix for voltages.
    TODO: Remove method
    :param A: The transmission matrix.
    :param R: The resistance value at the load.
    """
    au = np.zeros((3, 3), dtype='complex_')
    for i in range(0, 3):
        j = 2 * i
        k = j + 1
        Ai = A[[j, k], :][:, [j, k]]
        a = Ai[0, 0]
        b = Ai[0, 1]
        c = Ai[1, 0]
        d = Ai[1, 1]
        au[i, i] = (a * d - b * c) / (d - b / R)

    return au