
import numpy as np

r_1 = 2e-3                # [m]: inner radius (wire)
r_2 = 3.5e-3              # [m]: outer radius (shell)
depth = 300e-3            # [m]: Depth of wire (lz)
model_name = "wire"       # str : model name

I = 16                    # [A]   : applied current
J_0 = I/(np.pi*r_1**2)    # [A/m] : current density in the wire
mu_0 = 4*np.pi*1e-7       # [H/m] : permeability of vacuum (and inside of wire)
mu_shell = 5*mu_0


def H_phi(r):
    """
    Analytic solution of Magnetic Field

    Parameters
    ----------
    r : np.ndarray
        radius in [m]

    Returns
    -------
    h_phi : np.ndarray
        Magnetic field strength in [A/m]:
    """

    def H_phi_i(r):
        return J_0 / 2 * r

    def H_phi_a(r):
        return J_0 / 2 * r_1 ** 2 / r

    condition = r < r_1
    return condition * H_phi_i(r) + (~condition) * H_phi_a(r)

def B_phi(r):
    """
    Analytic solution of Magnetic Field

    Parameters
    ----------
    r : np.ndarray
        radius in [m]

    Returns
    -------
    B_phi : np.ndarray
        Magnetic field strength in [A/m]:
    """

    def B_phi_i(r):
        return J_0 * mu_0 / 2 * r

    def B_phi_a(r):
        return J_0 * mu_shell / 2 * r_1 ** 2 / r

    condition = r < r_1
    return condition * B_phi_i(r) + (~condition) * B_phi_a(r)


def A_z(r):
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
        return -I / (2 * np.pi) * (mu_0 / 2 * (r ** 2 - r_1 ** 2) / r_1 ** 2 + mu_shell * np.log(r_1 / r_2))

    def A_z_a(r):
        return -mu_shell * I / (2 * np.pi) * np.log(r / r_2)

    condition = r < r_1
    return condition * A_z_i(r) + (~condition) * A_z_a(r)

def W_magn():
    # magnetic energy(analytical)
    return I ** 2 * depth / (4 * np.pi) * (mu_0 / 4 + mu_shell * np.log(r_2 / r_1))


def Inductance():
    #[H]   : inductance (analytical)
    W_magn_analytic = W_magn()
    return 2 * W_magn_analytic / I ** 2
