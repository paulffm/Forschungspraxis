import gmsh
from matplotlib import pyplot as plt
from Mesh import Mesh
import numpy as np
from matplotlib.tri import Triangulation

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
    '''
    :param msh:
    :return:
    '''
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
    fig = plt.figure()
    plt.title('2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.triplot(triang)
    #ax1 = fig.add_subplot(121)
    #ax1.set_aspect('equal')
    # plots triangle grid
    #tpc = ax1.triplot(triang)

    '''# And here with z values
    z = (np.cos(x) * np.cos(3 * y)).flatten()
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    tpc = ax2.tripcolor(triang, z, shading='flat')
    fig.colorbar(tpc)
    plt.title('directly 2D')'''
    plt.show()

def plot_regions_of_mesh(msh, physical_groups):
    '''
    :param msh:
    :param physical_groups:
    :return:
    '''
    x = np.array(msh.node[:, 0], ndmin=1).T
    y = np.array(msh.node[:, 1], ndmin=1).T
    z = np.zeros_like(x)


    ## Visualize the regions of the model
    # Indices of elements in shell and wire
    elem_in_shell = entity_in_physical_group(physical_groups, msh.elem_to_node, 'SHELL')
    elem_in_wire = entity_in_physical_group(physical_groups, msh.elem_to_node, 'WIRE')

    # Indices of edges on ground
    edges_on_ground = entity_in_physical_group(physical_groups, msh.edge_to_node, 'GND')
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
    '''
    :param msh:
    :param reluctivity_in_elements:
    :return:
    '''

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
    '''
    :param msh:
    :param current_density_in_elems:
    :return:
    '''
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
    '''
    :param msh:
    :param a:
    :return:
    '''
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
    '''
    :param msh:
    :param b_field_abs:
    :return:
    '''
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