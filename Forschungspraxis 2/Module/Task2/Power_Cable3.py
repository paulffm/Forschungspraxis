import numpy as np
import matplotlib.pyplot as plt
from Power_Cable import PowerCable
show_plot = False

def main():

    k = [1, 2, 3]
    X_mag = []
    K_list = []
    a_mag = []
    current_list = []

    for i in k:
        power_cable = PowerCable()
        problem, shape_function = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False, k=i, type='magn')
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

    X_elec = []
    u_elec = []
    charge_list = []
    for i in k:
        power_cable = PowerCable()
        problem, shape_function = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False, k=i, type='elec')
        load = shape_function.load_vector(problem.regions, problem.excitations)
        X = load / power_cable.charge

        # ValueError: Matrix A is singular, because it contains empty row(s) -> without Reluktanz
        solution = problem.solve()
        mesh = problem.mesh
        curlcurl_matrix = solution.curlcurl_matrix

        print('Energy', solution.energy)
        X_elec.append(np.asarray(X.toarray()))
        K_list.append(np.asarray(curlcurl_matrix))
        u_elec.append(np.asarray(solution.vector_potential).reshape(-1, 1))
        charge_list.append(power_cable.charge)



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

    Xe_arr = np.concatenate((X_elec[0], X_elec[1], X_elec[2]), axis=1)
    ue_arr = np.concatenate((a_elec[0], a_elec[1], a_elec[2]), axis=1)






if __name__ == '__main__':
    main()