import gmsh
from matplotlib import pyplot as plt
from Meshsim import Mesh
from shape_function import ShapeFunction_N
import analytic_sol
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.sparse.linalg as las
from scipy.sparse import csr_matrix, csr_array, spmatrix
import plot_properties



def X(physical_groups, shape_function, mesh: Mesh) -> spmatrix:
    """The current distribution matrix X of size (N,1)."""
    x = np.zeros(mesh.num_elements * 3)
    rows = np.zeros(mesh.num_elements * 3)
    cols = np.zeros(mesh.num_elements * 3)  # TODO: Update cols if 2nd conductor is present.

    idx = entity_in_physical_group(physical_groups, mesh.elem_to_node, 'WIRE')  # The indices of wire elements with current I
    S: float = np.sum(shape_function.element_area[idx]) * 1  # The surface area of the wire

    for i, nodes in enumerate(mesh.elements):
        rows[i * 3:(i + 1) * 3] = nodes
        #x[i * 3:(i + 1) * 3] = shape_function.element_area[i] / (3*S) * np.ones(3) * idx[i]
        x[i * 3:(i + 1) * 3] = X_e(i, S, shape_function) * idx[i]

    return csr_matrix((x, (rows, cols)), shape=(mesh.num_node, 1))


def X_e(elem: int, S: float, shape_function) -> np.ndarray:
    """The local 3x1 current distribution matrix X.
    :param elem: The element for the local matrix.
    :param S: The surface area of the entire region.
    """
    return shape_function.element_area[elem] / (3*S) * np.ones(3)


def j_grid(I, physical_groups, shape_function, mesh: Mesh) -> spmatrix:
    """The grid-current right hand side vector of size (N,1)."""
    return X(physical_groups, shape_function, mesh) * I

def entity_in_physical_group(physical_group_data: dict, entity2node: np.ndarray, identifier):
    """
    Computes the indices of all entities that are in the physical group
    specified by the identifier

    Parameters
    ----------
    physical_group_data : dict
        dict with physical groups. Key is the ID of the group and value is a tuple with (dimension, name, indices of all nodes)
    entity2node : np.ndarray
        (K,N) array with K entities and N nodes per entity
    identifier : int or str
        Identifier of the physical group. The ID or the name of the group are accepted

    Returns
    -------
    entity_in_physical_group : np.ndarray
        (M,) array. With M being the number of entities in the physical group
    """
    if type(identifier) is str:
        for p in physical_group_data.keys():
            if physical_group_data[p][1] == identifier:
                identifier = p

    out = -1 * np.ones(entity2node.shape[0], dtype=int)
    for k in range(entity2node.shape[0]):
        if np.isin(entity2node[k, :], physical_group_data[identifier][2]).all():
            out[k] = k

    return out[out != -1]

def Knu_for_elem(k, shape_function, reluctivity_in_elements):
    '''
    :param k:
    :param shape_function:
    :param reluctivity_in_elements:
    :return:
    '''

    # integral(v rot(Wi) * rot(Wj) dV) =>
    # v (Wi dx,
    # ((b.T * b + c.T * c ) / (4 * area * l_z)) * reluctivity
    # produces 3x172 * 172x3 = 3x3 output
    grad_Ni_grad_Nj = (np.array(shape_function.b[k, :], ndmin=2).T @ np.array(shape_function.b[k, :], ndmin=2) +
           np.array(shape_function.c[k, :], ndmin=2).T @ np.array(shape_function.c[k, :], ndmin=2)) \
                      / (4*shape_function.element_area[k]*shape_function.depth)
    return reluctivity_in_elements[k] * grad_Ni_grad_Nj


def main():
    r_1 = 2e-3  # [m]: inner radius (wire)
    r_2 = 3.5e-3  # [m]: outer radius (shell)
    depth = 300e-3  # [m]: Depth of wire (lz)
    model_name = "wire"  # str : model name
    I = 16  # [A]   : applied current
    J_0 = I / (np.pi * r_1 ** 2)  # [A/m] : current density in the wire
    mu_0 = 4 * np.pi * 1e-7  # [H/m] : permeability of vacuum (and inside of wire)
    mu_shell = 5 * mu_0
    eps_0 = 8.8541878128 * 1e-12


    # plot of Analytic Solution:
    '''r_list = np.linspace(0, r_2, 100)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(r_list, analytic_sol.B_phi(r_list))
    axs[0].set_title("B")
    axs[0].set(xlabel='r', ylabel='B')
    axs[1].plot(r_list, analytic_sol.H_phi(r_list))
    axs[1].set_title("H")
    axs[1].set(xlabel='r', ylabel='H')
    axs[2].plot(r_list, analytic_sol.A_z(r_list))
    axs[2].set_title("A")
    axs[2].set(xlabel='r', ylabel='A')
    plt.show()'''


    ##### Task2: Construction of a finite-element model with Gmsh #####

    msh = Mesh.create()
    #print(msh.num_elements, msh.num_node): 172, 101 weniger elements
    pg = gmsh.model.getPhysicalGroups()
    # (dim, tag)
    # pg [(1, 3), (2, 1), (2, 2)]

    # Physical Groups
    # In every entry in physical_groups we define the following structure (TAG (dimension, name, indices of all nodes))
    # getNodesforPhysicalGroup(dim, tag)
    physical_groups = dict()

    ##### Task 3: Visualization of mesh and regions #####

    for group in pg:
        # physical_groups unterscheidung durch tags: zugriff: physical_groups[3]
        # getphysicalname(dim, tag) -> name as string 'GND'
        # getNodesforPhysicalGroup(dim, tag) -> node_tags, node_coord hier nur node_tags wegen [0];

        physical_groups[group[1]] = (group[0], gmsh.model.getPhysicalName(group[0], group[1]),
                                     gmsh.model.mesh.getNodesForPhysicalGroup(group[0], group[1])[0] - 1)

        # Dict: (Tag, (Dim, physicalName, array with node tags, array with coord of nodes)
        # 3 GND 1 WIRE 2 SHELL
        # access by tag physical_groups[1] oder [2] [3] dann ein ganzes dict
        # physical_groups[1][2] nur tags von wire
        # physical_groups[2][2] name von wire

    #print(physical_groups[1][2])
    gmsh.finalize()

    # plot msh: directly 2D plot,colorbar just for a z Value
    print(msh.num_elements)
    plot_properties.plot_mesh(msh)

    # plot regions of mesh
    plot_properties.plot_regions_of_mesh(msh, physical_groups)

    # indices for all entities by physical group
    elem_in_shell = entity_in_physical_group(physical_groups, msh.elem_to_node, 'SHELL')
    elem_in_wire = entity_in_physical_group(physical_groups, msh.elem_to_node, 'WIRE')

    # Compute element-wise reluctivity: shell and inside of wire
    # Permeabilität = Durchlässigkeit Magnetfeld; Reluktanz=magn. Widerstand
    # magn. Fluss folgt dem Weg mit geringstem magn. Widerstand
    reluctivity_in_elem = 1 / mu_0 * np.ones(msh.num_elements)  # [m/H] : reluctivities, one per element
    reluctivity_in_elem[elem_in_shell] = 1 / mu_shell  # [m/H] : reluctivities for the iron shell

    # plot reluctivity
    plot_properties.plot_reluctivity(msh, reluctivity_in_elem)

    # Task 4: setup the FE shape functions and assemble the stiffness matrix and load vector.
    # construct shape_function

    x1 = msh.node[msh.elem_to_node, 0]  # x coordinates of the nodes for each element (num_nodes x 1)
    y1 = msh.node[msh.elem_to_node, 1]  # y coordinates of the nodes for each element (num_nodes x 1)

    x2 = np.roll(x1, -1, axis=1)
    y2 = np.roll(y1, -1, axis=1)
    x3 = np.roll(x1, -2, axis=1)
    y3 = np.roll(y1, -2, axis=1)

    # Definition of coefficients according to HDG:
    # and are equivalent: all (num_elem x 3):
    # für jeden Punkt die Node function
    a = x2 * y3 - x3 * y2
    b = y2 - y3
    c = x3 - x2

    # mean weil für alle gleich ist: also aus 172 x 3 -> 172 x 1
    element_area = np.mean(((x2 * y3 - x3 * y2) + (y2 - y3) * x1 + (x3 - x2) * y1) / 2, 1)

    # Instantiation of Shape function: gives only object
    shape_function = ShapeFunction_N(depth, element_area, a, b, c)

    # Assign Knu for global indices: exactly taken from supporting remarks
    index_rows = np.zeros((9 * msh.num_elements), dtype='int')
    index_columns = np.zeros((9 * msh.num_elements), dtype='int')
    elementwise_entries = np.zeros((9 * msh.num_elements))

    # loop over elements
    for k in range(msh.num_elements):

        # get global indices for nodes for element k: (3, ): Wie zeile
        global_indices = msh.elem_to_node[k, :]

        # (3, 3) -> zeile dreimal untereinander
        triple_global_indices = np.array([global_indices, global_indices, global_indices])

        # np.reshape(triple_global_indices, (9)) schreibt in eine Zeile: (9, )
        # schreibt nach und nach alle 9 indices nebeneinander und zwar wie folgt:
        # immer 3 mal nebeinander: index_row [44 58 49 44 58 49 44 58 49 ...],
        index_rows[9 * k:9 * k + 9] = np.reshape(triple_global_indices, (9))

        # dann col: [44 44 44 58 58 58 49 49 49  ..]
        index_columns[9 * k:9 * k + 9] = np.reshape(triple_global_indices.T, (9))

        # lokales Knu: ((b.T * b + c.T * c ) / (4 * area * l_z)) * reluctivity -> produziert 3x3 output
        #print(Knu_for_elem(k, shape_function, reluctivity_in_elements))

        # aus 3x3 wird 1x9 und das in elementwise_entries geschrieben
        elementwise_entries[9 * k:9 * k + 9] = np.reshape(Knu_for_elem(k, shape_function, reluctivity_in_elem), (9))

    # Assembly of Knu
    index_rows = index_rows.T
    index_columns = index_columns.tolist()
    elementwise_entries = elementwise_entries.tolist()

    # Build Knu Matrix: # [1/H] : circuit-reluctance matrix (num_nodes x num_nodes array but sparse)
    Knu = csr_matrix((elementwise_entries, (index_rows, index_columns)))
    print('Knu shape', Knu, Knu.shape)

    # Print structure of matrix: dünn besetzt
    plt.spy(Knu, markersize=1)
    plt.show()

    ##### Task 4: Define load vector j_grid, element-wise current density #####
    # Set element-wise current density in wire region
    j_in_elems = np.zeros(msh.num_elements)


    # [A/m^2]: current density: I/A von jedem element: Nur in Wire sonst ja isoliert
    # np.sum(shape_function.element_area[elem_in_wire]) surface area of wire

    # current_density_elems: jedes Wire dreieck gleichen Anteil an I
    j_in_elems[elem_in_wire] = I / np.sum(shape_function.element_area[elem_in_wire])


    #print('elem_are', np.sum(shape_function.element_area * current_density_in_elems) == I)
    'grid currents_e = J integral(Ne,i dA) = J Ae/3 -> Ni aufintegriert über Fläche für jedes Element'
    grid_currents = j_in_elems * shape_function.element_area / 3
    x_elems = grid_currents / I


    # (172, 3): grid_currents:
    # 3 mal nebeneinander kopiert mit np.tile um für jeden Node die Contribution zu bestimmen
    grid_currents = np.tile(grid_currents, (3, 1)).transpose()
    x_elems = np.tile(x_elems, (3, 1)).transpose()

    # vector with values for current contribution of each element on the nodes.
    values = np.zeros(msh.num_node)
    x_values = np.zeros(msh.num_node)

    # Iteration durch jedes Element i: und wenn Node k an Ecke von Element i:
    # Addiere Beitrag Ii,k zu Node k in Vector values:
    # Am Ende: Für jeden Node stehen dort die addierten Strombeiträge von jedem Element, in welchem sich Node k befindet
    for k in range(0, 3):
        for i in range(0, msh.num_elements - 1):
            idx_node = msh.elem_to_node[i, k]
            values[idx_node] += grid_currents[i, k]
            x_values[idx_node] += x_elems[i, k]

    # Assembly of grid current vector: (num_nodes, 1) array but sparse, with current on each node
    j_grid = csr_matrix((values, (np.arange(msh.num_node), np.zeros(msh.num_node))), shape=(msh.num_node, 1))
    x_grid = csr_matrix((x_values, (np.arange(msh.num_node), np.zeros(msh.num_node))), shape=(msh.num_node, 1))
    print('j', j_grid, j_grid.shape)
    print('x', x_grid, x_grid * I == j_grid)

    # plot current density in elements
    plot_properties.plot_current(msh, j_in_elems)

    print('unit of Knu: [1/H] : circuit-reluctance matrix')
    print('unit of load vector: [A/m^2]: current density')

    ##### Task 5: First validation is the check of the magnetic energy #####

    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T

    # Radial coordinate in [m]:
    r = np.sqrt(x ** 2 + y ** 2)

    # Analytische Lösung von A in [Tm]:
    A_analytic = depth * analytic_sol.A_z(r)

    # Magnetische Energie (Analytisch und numerisch) in [J]
    # Gesamtenergie eines magnetostatischen Feldes: W = 0.5 Integral (A *J) dV= 0.5 * Integral (A * K * A) dV (KA=J)
    W_magn_test = 1 / 2 * A_analytic @ Knu @ A_analytic
    W_magn_analytic = analytic_sol.W_magn()

    print('Magnetic energy (analytic solution)           :', W_magn_analytic, 'J')
    print('Magnetic energy (analytic Az, numerical Knu)  :', W_magn_test, 'J')

    ##### Task 6: setup and solve the magnetostatic problem #####

    a = np.zeros((msh.num_node, 1))  # Initialize vector of dofs

    # indices of GND are the boundary:
    idx_bc = physical_groups[3]  #
    idx_bc = idx_bc[2]  # take only the indices out of dict: 28
    #
    value_bc = np.zeros((len(idx_bc), 1))
    # the indices where nothing is given: indices where we have to calculate a: 73 -> num_nodes - dof = index_constraint
    # restlichen indizes -> DoF: An diesen muss a berechnet werden
    idx_dof = np.setdiff1d(np.arange(msh.num_node), idx_bc).tolist()

    # Reduce the system: Knu (num_nodes x num_nodes) --> (num_dof x num_dof): Remove the BC indizes -> known entries
    Knu_red = Knu[idx_dof, :]
    Knu_red = Knu_red[:, idx_dof]

    j_grid_dof = j_grid[idx_dof]
    rhs = j_grid_dof

    # Solve the system: Ka = j spsolver: Ax=b
    a_shrink = spsolve(Knu_red, rhs)

    # Inflate A back to full size: num_nodes x 1
    a_shrink = np.array(a_shrink, ndmin=2).T  # (73, ) only non boundary nodes
    a[idx_dof] = a_shrink  # filled to (101,1)
    a[idx_bc] = value_bc
    a = a.reshape(len(a))  # (101) for every node

    # example calculation L

    # wie L?
    x_grid_shrink = x_grid[idx_dof]
    x_hat = np.zeros((msh.num_node, 1))
    x_hat_shrink = spsolve(Knu_red, x_grid_shrink)
    x_hat[idx_dof] = x_hat_shrink.reshape(-1, 1)
    x_hat[idx_bc] = value_bc.reshape(-1, 1)
    x_hat = np.asarray(x_hat).reshape(-1, 1)
    print(x_grid.shape, x_hat.shape)

    L_fe = x_grid.T @ x_hat
    print('Analytical L', analytic_sol.Inductance())
    print('FE L', L_fe, L_fe.shape, x_grid.T @ (a / I))





    # plot sol: on ground = 0
    plot_properties.plot_sol(msh, a)

    ##### Task 7: Calculate magnetic flux density #####
    #
    # bx = sum(c * A / 2 * area) / l_z , by = sum(b * A / 2 * area)  / (l_z)
    # 172 x 2
    b_field = np.vstack([np.sum(shape_function.c * a[msh.elem_to_node[:]] / (2 * shape_function.element_area[:, None]), 1)
                         / shape_function.depth,
                         - np.sum(shape_function.b * a[msh.elem_to_node[:]] / (2 * shape_function.element_area[:, None]), 1)
                         / shape_function.depth]).T

    # num_elements x 1
    b_field_abs = np.linalg.norm(b_field, axis=1) # [T]    : magnitude of the magnetic flux density

    # plot b_field:
    # man sieht: ist stärker, wo Reluktanz geringer ist -> in Shell ist B folglich größer
    plot_properties.plot_bfield(msh, b_field_abs)


    # Compare Results: magnetic energy in [J]
    W_magn_fe = 1 / 2 * a @ Knu @ a
    W_magn_fe2 = np.sum(1 / 2 * reluctivity_in_elem * b_field_abs ** 2
                        * shape_function.element_area * shape_function.depth)
    print('Validity Check:')
    print('Magnetic energy (Analytisch)                :', W_magn_analytic, 'J')
    print('Magnetic energy (FE)                      :', W_magn_fe, 'J')
    print('Magnetic energy (Integrierte FE-LSG)   :', W_magn_fe2, 'J')

    # rel error
    rel_error = np.abs((W_magn_fe - W_magn_analytic) / W_magn_analytic)
    print(f'Relative error of energy: {rel_error}')
    # conv_order = (np.log(rel_error1) -  np.log(rel_error2)) / np.log(size1) - np.log(size2))

    ##### Task 8, 9: relativ error of energy and convergence study in pyrit script#####
    # in pyrit
if __name__ == '__main__':
    main()

