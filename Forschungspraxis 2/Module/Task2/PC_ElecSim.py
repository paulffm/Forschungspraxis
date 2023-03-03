import numpy as np
import matplotlib.pyplot as plt
from Power_Cable import PowerCable

show_plot = False

def main():

    k = [1, 2, 3]

    X_elec = []
    u_elec = []
    K_list = []
    charge_list = []

    for i in k:
        power_cable = PowerCable()
        problem, shape_function = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False, k=i, type='elec')
        load = shape_function.load_vector(problem.regions, problem.excitations)
        X = load / power_cable.charge

        # ValueError: Matrix A is singular, because it contains empty row(s) -> without Reluktanz
        solution = problem.solve()
        mesh = problem.mesh
        divgrad_matrix = solution.divgrad_matrix

        print('Energy', solution.energy)
        X_elec.append(np.asarray(X.toarray()))
        K_list.append(np.asarray(divgrad_matrix))
        u_elec.append(np.asarray(solution.potential).reshape(-1, 1))
        charge_list.append(power_cable.charge)
        print('pot', solution.potential, solution.potential.shape)



        if show_plot:
            solution.plot_energy_density()
            plt.show()


        # Compute the magnetic flux density magnitude
        b_abs = np.linalg.norm(solution.d_field, axis=1)

        # Plots the magnetic flux density, style options are 'arrows', 'abs', 'stream'.
        if show_plot:
            solution.plot_d_field('abs')
            #solution.plot_d_field('arrows')
            #solution.plot_d_field('stream')
            solution.plot_equilines()
            solution.plot_potential()
            solution.plot_e_field('abs')
            #solution.plot_load_vector()
            plt.show()

    Xe_arr = np.concatenate((X_elec[0], X_elec[1], X_elec[2]), axis=1)
    ue_arr = np.concatenate((u_elec[0], u_elec[1], u_elec[2]), axis=1)

    r_w = 1.1e-3
    sigma = 57.7e6

    C_inv = Xe_arr.T @ ue_arr / charge_list[0]
    C = np.linalg.inv(C_inv)
    print('Capacity', C)
    #print('C2', charge_list[0] * np.linalg.inv(Xe_arr.T @ ue_arr))


if __name__ == '__main__':
    main()