import numpy as np
from typing import NoReturn
from dataclasses import dataclass
from scipy.constants import constants
from scipy.sparse import linalg, dia_matrix
from tasks.BoxResonator import BoxResonator


@dataclass
class AnalyticalSolution:
    width_x: float
    width_y: float
    width_z: float
    permeability: float = constants.mu_0
    permittivity: float = constants.epsilon_0

    def k_x(self, n_x: int) -> float:
        return n_x * np.pi / self.width_x

    def k_y(self, n_y: int) -> float:
        return n_y * np.pi / self.width_y

    def k_z(self, n_z: int) -> float:
        return n_z * np.pi / self.width_z

    def eigenfrequencies(self, n_x: int, n_y: int, n_z: int) -> float:
        if (n_x == 0 and (n_y == 0 or n_z == 0)) or (n_y == 0 and n_z == 0):
            return 0
        return 1 / (2 * np.pi) * np.sqrt((self.k_x(n_x)**2 + self.k_y(n_y)**2 + self.k_z(n_z)**2) / (self.permeability *
                                                                                                     self.permittivity))


def compute_eigenfrequencies(sides_box: np.ndarray, mesh_size: np.ndarray, num_eigf: int = 15) -> np.ndarray:
    # create model:
    box = BoxResonator(sides_box)
    box.set_mesh(mesh_size)

    # get operator:
    reluctance_dg, capacitance_pg, curl_operator = box.get_operators()
    curl_reluctance_curl_matrix = curl_operator.T * np.diag(reluctance_dg.flatten()) @ curl_operator

    # solve eigenvalue equation:
    # curl_reluctance_curl_matrix * grid_voltage = omega**2 capacitance_pg * grid_voltage
    omega_squared_init = (2 * np.pi * 1.5e9) ** 2
    omega_squared, _ = linalg.eigs(A=curl_reluctance_curl_matrix, M=np.diag(capacitance_pg.flatten()), k=num_eigf,
                                   sigma=omega_squared_init)

    return np.real(np.sqrt(omega_squared)) / 2 / np.pi


def main():
    # data:
    sides_box = np.array([0.200, 0.160, 0.120])  # [m] : domain sizes (x, y, z)
    mesh_sizes = [np.array([.04, .04, .04]), np.array([.02, .02, .02]), np.array([.01, .01, .01])]  # [m]
    num_eig_freq = 15  # number of considered eigenvalues

    # numerically computed eigenfrequencies:
    eig_freq = np.zeros((len(mesh_sizes), num_eig_freq))
    for k, mesh_size in enumerate(mesh_sizes):
        eig_freq[k, :] = compute_eigenfrequencies(sides_box, mesh_size)

    # get analytical eigenfrequencies:
    from task3 import analytical_eigenfreq
    eig_freq_ana = analytical_eigenfreq(*sides_box)[:15]

    a = 1


if __name__ == "__main__":
    main()

