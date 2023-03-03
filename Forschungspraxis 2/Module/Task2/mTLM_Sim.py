import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mTLM import Z, Y, decompose, Z_ch, beta_m, A_m, A, solve, P, Pout, Ak_m
from matplotlib.pyplot import cm
from Power_Cable import PowerCable

show_plot = True


def main():

    f = 1e3
    l = 1
    R = np.eye(3) * 1e-3
    G = np.zeros((3, 3))
    L = np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]]) * 1e-6
    C = np.array([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]]) * 1e-9

    Z_z = Z(R, L, f)
    Y_y = Y(G, C, f)

    Zm, Ym, Tu, Ti = decompose(Z_z, Y_y)
    np.savetxt('Ti.csv', Ti)

    k = [1, 2, 3]
    X_mag = []
    K_list = []
    a_mag = []
    current_list = []

    for i, k_i in enumerate(k):
        power_cable = PowerCable(current=Ti[:, i])
        #power_cable = PowerCable(current=1)
        problem, shape_function = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False, k=k_i,
                                                             type='magn', exci_type=3)
        load = shape_function.load_vector(problem.regions, problem.excitations)
        X = load / power_cable.current

        # ValueError: Matrix A is singular, because it contains empty row(s) -> without Reluktanz
        solution = problem.solve()
        mesh = problem.mesh
        curlcurl_matrix = solution.curlcurl_matrix

        print('Energy', solution.energy)
        X_mag.append(np.asarray(X.toarray()))
        K_list.append(np.asarray(curlcurl_matrix))
        a_mag.append(np.asarray(solution.vector_potential).reshape(-1, 1))
        current_list.append(power_cable.current)

        if show_plot:
            solution.plot_energy_density()
            plt.show()

        # Compute the magnetic flux density magnitude
        b_abs = np.linalg.norm(solution.b_field, axis=1)

        # Plots the magnetic flux density, style options are 'arrows', 'abs', 'stream'.
        if show_plot:
            solution.plot_b_field('abs')
            solution.plot_b_field('arrows')
            solution.plot_b_field('stream')
            solution.plot_equilines()
            plt.show()

    Xm_arr = np.concatenate((X_mag[0], X_mag[1], X_mag[2]), axis=1)
    am_arr = np.concatenate((a_mag[0], a_mag[1], a_mag[2]), axis=1)

    r_w = 1.1e-3
    sigma = 57.7e6

    R = np.eye(3) * (1 / (sigma * np.pi * r_w ** 2))
    print('Resistance:', R)
    L = Xm_arr.T @ am_arr / current_list[0]
    print('Inductivity', L)



if __name__ == '__main__':
    main()
