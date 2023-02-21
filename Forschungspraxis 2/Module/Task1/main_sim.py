import gmsh
import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.sparse import csr_matrix
from Meshsim import Mesh
from matplotlib.tri import Triangulation
from shape_function import ShapeFunction_N
import analytic_sol
from scipy.sparse.linalg import spsolve

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


def plot_mesh(msh):
    # %% Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=0, elev=90)

    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    z = np.zeros_like(x)
    ax.plot_trisurf(x, y, z)
    plt.title('3D Projection')
    plt.show()

    # Alternative plotting. Here the result is directly 2D.
    # Triangulation(coords, indices of triangles)
    triang = Triangulation(x, y, msh.elem_to_node)
    print(triang)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal')
    # plots triangle grid
    tpc = ax1.triplot(triang)

    # And here with z values
    z = (np.cos(x) * np.cos(3 * y)).flatten()
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    tpc = ax2.tripcolor(triang, z, shading='flat')
    fig.colorbar(tpc)
    plt.title('directly 2D')
    plt.show()

def plot_regions_of_mesh(msh, physical_groups):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    z = np.zeros_like(x)


    ## Visualize the regions of the model
    # Indices of elements in shell and wire
    elem_in_shell = entity_in_physical_group(physical_groups, msh.elem_to_node, 'SHELL')
    elem_in_wire = entity_in_physical_group(physical_groups, msh.elem_to_node, 'WIRE')

    # Indices of edges on ground
    edges_on_ground = entity_in_physical_group(physical_groups, msh.edge_to_node, 'GND')

    # Instead of using the name of the physical group, one can also use its ID:
    # elem_in_shell = entity_in_physical_group(physical_groups, msh.elem_to_node, 2)
    # elem_in_wire = entity_in_physical_group(physical_groups, msh.elem_to_node, 1)
    # edges_on_ground = entity_in_physical_group(physical_groups, edge_to_node, 3)

    triang_shell = Triangulation(x, y, msh.elem_to_node[elem_in_shell])
    triang_wire = Triangulation(x, y, msh.elem_to_node[elem_in_wire])

    # Plot shell, wire and edges on ground potential
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.triplot(triang_wire, color='orange', label='wire')
    ax.triplot(triang_shell, color='blue', label='shell')
    plt.legend()
    for edge in edges_on_ground:
        # in i zeichnet von line i von x zu y koordinate
        node_x = msh.node[msh.edge_to_node[edge, [0, 1]], 0]
        node_y = msh.node[msh.edge_to_node[edge, [0, 1]], 1]
        line, = ax.plot(node_x, node_y, color='red')
    line.set_label('ground')
    plt.title('Regions plot')
    plt.legend()
    plt.show()

def plot_reluctivity(msh, reluctivity_in_elements):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    triang = Triangulation(x, y, msh.elem_to_node)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tpc = ax.tripcolor(triang, facecolors=reluctivity_in_elements)
    fig.colorbar(tpc)
    ax.triplot(triang, color='black', lw=0.1)
    ax.set_aspect('equal', 'box')
    plt.title('Reluctivity')
    plt.show()


def plot_current(msh, current_density_in_elems):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    triang = Triangulation(x, y, msh.elem_to_node)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tpc = ax.tripcolor(triang, facecolors=current_density_in_elems)
    fig.colorbar(tpc)
    ax.triplot(triang, color='black', lw=0.1)
    ax.set_aspect('equal', 'box')
    plt.title('current density')
    plt.show()

def plot_sol(msh, a):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    triang = Triangulation(x, y, msh.elem_to_node)
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    surf = ax.tripcolor(triang, a, cmap='viridis')  # cmap=plt.cm.CMRmap)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('solution for a')
    plt.show()

def plot_bfield(msh, b_field_abs):
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    triang = Triangulation(x, y, msh.elem_to_node)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tpc = ax.tripcolor(triang, facecolors=b_field_abs)
    fig.colorbar(tpc)
    ax.triplot(triang, color='black', lw=0.1)
    ax.set_aspect('equal', 'box')
    plt.title('magnetic flux density')
    plt.show()

def Knu_for_elem(k, shape_function, reluctivity_in_elements):
    """Local matrix"""

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
    plot_mesh(msh)

    # plot regions of mesh
    plot_regions_of_mesh(msh, physical_groups)

    # indices for all entities by physical group
    elem_in_shell = entity_in_physical_group(physical_groups, msh.elem_to_node, 'SHELL')
    elem_in_wire = entity_in_physical_group(physical_groups, msh.elem_to_node, 'WIRE')

    # Compute element-wise reluctivity: shell and inside of wire
    # Permeabilität = Durchlässigkeit Magnetfeld; Reluktanz=magn. Widerstand
    # magn. Fluss folgt dem Weg mit geringstem magn. Widerstand
    reluctivity_in_elements = 1 / mu_0 * np.ones(msh.num_elements)  # [m/H] : reluctivities, one per element
    reluctivity_in_elements[elem_in_shell] = 1 / mu_shell  # [m/H] : reluctivities for the iron shell

    # plot reluctivity
    plot_reluctivity(msh, reluctivity_in_elements)

    # Task 4: setup the FE shape functions and assemble the stiffness matrix and load vector.
    # construct shape_function

    x1 = msh.node[msh.elem_to_node, 0]  # x coordinates of the nodes for each element
    y1 = msh.node[msh.elem_to_node, 1]  # y coordinates of the nodes for each element
    print(x1.shape, x1[:10]) # 172, 3->jedes Dreieck 3 Punkte mit 3 x Koordinaten
    # roll rotiert x, y koordinaten um 1, 2 Stellen:

    # um die 3 Nodal functions in jedem Dreieck zu berechnen: (immer zwischen 2 Punkten)
    # zwischen P1, P2; P1, P3 und P2, P3
    x2 = np.roll(x1, -1, axis=1)
    y2 = np.roll(y1, -1, axis=1)
    x3 = np.roll(x1, -2, axis=1)
    y3 = np.roll(y1, -2, axis=1)
    print('x3', x3[:10])

    # Definition of coefficients according to HDG:
    # and are equivalent: all (num_elem x 3): je nach dem, was als Punkt 1, 2,3 anderes a, b, c
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

        # get global indices for nodes for element k: (3, ): Zie zeile
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
        elementwise_entries[9 * k:9 * k + 9] = np.reshape(Knu_for_elem(k, shape_function, reluctivity_in_elements),(9))

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
    current_density_in_elems = np.zeros(msh.num_elements)  # [A/m^2]: current density

    # [A/m^2]: current density: I/A von jedem element: Nur in Wire sonst ja isoliert
    current_density_in_elems[elem_in_wire] = I / np.sum(shape_function.element_area[elem_in_wire])

    # localized current density, grid current [A]: (172,)
    print('elem_are', shape_function.element_area)
    grid_currents = current_density_in_elems * shape_function.element_area / 3

    # inflate for nodal accumulation: (172, 3): grid_currents: 3 mal nebeneinander kopiert mit np.tile
    grid_currents = np.tile(grid_currents, (3, 1)).transpose()

    # vector with values for current contribution of each element on the nodes.
    values = np.zeros(msh.num_node)

    # Iteration durch jedes Element i: und wenn Node k an Ecke von Element i:
    # Addiere Beitrag Ii,k zu Node k in Vector values:
    # Am Ende: Für jeden Node stehen dort die addierten Strombeiträge von jedem Element, in welchem sich Node k befindet
    for k in range(0, 3):
        for i in range(0, msh.num_elements - 1):
            idx_node = msh.elem_to_node[i, k]
            values[idx_node] += grid_currents[i, k]

    # Assembly of grid current vector: (num_nodes, 1) array but sparse, with current on each node
    j_grid = csr_matrix((values, (np.arange(msh.num_node), np.zeros(msh.num_node))), shape=(msh.num_node, 1))
    print('j', j_grid, j_grid.shape)




    # plot current density in elements
    plot_current(msh, current_density_in_elems)

    print('unit of Knu: [1/H] : circuit-reluctance matrix')
    print('unit of load vector: [A/m^2]: current density')

    ##### Task 5: First validation is the check of the magnetic energy #####

    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T

    # [m]    : radial coordinate
    r = np.sqrt(x ** 2 + y ** 2)

    # [Tm]   : analytic solution for the magnetic vector potential projected onto the mesh
    A_analytic = depth * analytic_sol.A_z(r)

    # [J]    : magnetic energy (analytic solution, numerical circuit-reluctance matrix)
    # Gesamtenergie eines magnetostatischen Feldes: W = 0.5 Integral (A *J) dV= 0.5 * Integral (A * K * A) dV (KA=J)
    W_magn_test = 1 / 2 * A_analytic @ Knu @ A_analytic
    W_magn_analytic = analytic_sol.W_magn()

    print('Magnetic energy (analytic solution)           :', W_magn_analytic, 'J')
    print('Magnetic energy (analytic Az, numerical Knu)  :', W_magn_test, 'J')

    ##### Task 6: setup and solve the magnetostatic problem #####

    a = np.zeros((msh.num_node, 1))  # Initialize vector of dofs

    # Indices and values of nodes at Dirichlet boundary: indices of GND are the boundary
    index_constraint = physical_groups[3]  # [@]: edge indices of the shell boundary on GND
    index_constraint = index_constraint[2]  # strip indices: take only the indices out of dict: 28
    # [@]: indices of the nodes affected by homogeneous boundary condition
    value_constraint = np.zeros((len(index_constraint), 1))
    # the indices where nothing is given: indices where we have to calculate a: 73 -> num_nodes - dof = index_constraint
    index_dof = np.setdiff1d(np.arange(msh.num_node),
                             index_constraint).tolist()  # [@]: indices of the degrees of freedom

    # Shrink the system: Knu (num_nodes x num_nodes) --> (num_dof x num_dof)
    Knu_shrink = Knu[index_dof, :]  # remove boundary condition (=known) entries in Knu matrix
    Knu_shrink = Knu_shrink[:, index_dof]

    j_grid_dof = j_grid[index_dof]
    rhs = j_grid_dof
    # Solve the system: Ka = j spsolver: Ax=b
    a_shrink = spsolve(Knu_shrink, rhs)

    # Inflate
    a_shrink = np.array(a_shrink, ndmin=2).T  # (73, ) only non boundary nodes
    a[index_dof] = a_shrink  # filled to (101,1)
    a[index_constraint] = value_constraint
    a = a.reshape(len(a))  # (101) for every node


    # idx wire
    idx_wire = physical_groups[1]  # [@]: edge indices of the shell boundary on GND
    idx_wire = idx_wire[2]
    X_distr = np.zeros((msh.num_node, 1))
    X_distr[idx_wire] = 1

    # wie L?
    X_distr_shrink = X_distr[index_dof]
    x_hat = np.zeros((msh.num_node, 1))
    x_hat_shrink = spsolve(Knu_shrink, index_dof)
    x_hat[index_dof] = x_hat_shrink.reshape(-1, 1)
    x_hat[index_constraint] = value_constraint.reshape(-1, 1)
    x_hat = a.reshape(len(x_hat))
    print(X_distr.T * x_hat, analytic_sol.Inductance())
    L_hat = X_distr.T @ Knu @ X_distr
    print(L_hat)



    # plot sol: on ground = 0
    plot_sol(msh, a)

    ##### Task 7: Calculate magnetic flux density #####
    #
    # bx = sum(c * A / 2 * area) / l_z , by = sum(b * A / 2 * area)  / (l_z)
    # 172 x 2
    b_field = np.vstack([np.sum(shape_function.c * a[msh.elem_to_node[:]] / (2 * shape_function.element_area[:, None]), 1)
                         / shape_function.depth,
                         np.sum(shape_function.b * a[msh.elem_to_node[:]] / (2 * shape_function.element_area[:, None]), 1)
                         / shape_function.depth]).T
    print('b_field', b_field, b_field.shape)
    # 172
    b_field_abs = np.linalg.norm(b_field, axis=1) # [T]    : magnitude of the magnetic flux density
    print(b_field_abs, b_field_abs.shape)

    # plot b_field:
    # man sieht: ist stärker, wo Reluktanz geringer ist -> in Shell ist B folglich größer
    plot_bfield(msh, b_field_abs)


    # Validity check
    W_magn_fe = 1 / 2 * a @ Knu @ a  # [J]    : magnetic energy

    # W =
    W_magn_fe2 = np.sum(1 / 2 * reluctivity_in_elements * b_field_abs ** 2
                        * shape_function.element_area * shape_function.depth)  # [J]     : magnetic energy (integrated)
    print('Validity Check:')
    print('Magnetic energy (analytic solution)                :', W_magn_analytic, 'J')
    print('Magnetic energy (FE solution)                      :', W_magn_fe, 'J')
    print('Magnetic energy (numerical solution, integrated)   :', W_magn_fe2, 'J')

    rel_error = np.abs((W_magn_fe - W_magn_analytic) / W_magn_analytic)
    print(f'Relative error of energy: {rel_error}')
    # conv_order = (np.log(rel_error1) -  np.log(rel_error2)) / np.log(size1) - np.log(size2))

    ##### Task 8, 9: relativ error of energy and convergence study in pyrit script#####

    # to do: how to calculate B, Knu
if __name__ == '__main__':
    main()

