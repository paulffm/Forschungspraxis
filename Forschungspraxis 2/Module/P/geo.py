from matplotlib import pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
import gmsh

#msh = gmsh.model.mesh

def node_coords(group_tag, physical_groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the coordinates of the nodes of the mesh.
    :return: The x,y,z coordinates.
    """
    # node_tags und nodes für gnd, wire ,shell
    #(Tag, (Dim, physicalName, array with node tags, array with coord of nodes)
    node_tags = physical_groups[group_tag][2]
    nodes = physical_groups[group_tag][3]
    node_tags = node_tags - 1

    i = np.arange(0, len(node_tags))
    x = nodes[3 * i]
    y = nodes[3 * i + 1]
    z = nodes[3 * i + 2]
    return x, y, z

def plot_geometry(physical_groups):
    """
    Plots the cable geometry and visualizes the physical groups.
    """
    #wire, shell, gnd = cable()
    # node_coords takes dimension and tag: 1, 2, 3 -> findet koordinaten der Dreiecke, der verschiedenen Gruppen
    xw, yw, zw = node_coords(1, physical_groups) #wire
    xs, ys, zs = node_coords(2, physical_groups)
    xg, yg, _ = node_coords(3, physical_groups)

    # Creating plot
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(xw, yw, zw, color="blue")
    ax.plot_trisurf(xs, ys, zs, color="green", alpha=0.5)
    ax.plot(xg, yg, color="red")
    plt.show()

'''def element_node_coords(element_type: int, tag=-1) -> Dict[int, List[np.ndarray]]:
    """An element-tag to node coords dict.
    :param element_type: The type of elements to store in the dict.
    :param tag: The tag of elements to get.
    """

    element_tags, node_tags = gmsh.model.mesh.get_elements_by_type(element_type, tag)
    node_tags = np.array_split(node_tags, len(node_tags) / 3)
    coords: List[List[np.ndarray]] = list()

    for tags in node_tags:
        coords.append([gmsh.model.mesh.get_node(t)[0] for t in tags])

    return dict(zip(element_tags, coords))'''

'''def elem_2_node_coord(element_type=2):
    # dict with for every element tag the 9 point coordinates as list

    elem_tags, node_tags = gmsh.model.mesh.getElementsByType(2)
    node_tags = np.asarray(node_tags).reshape(-1, 3)

    #print(msh.elem_to_node) -> andere punkte
    #coord: List[List[np.ndarray]] = list()
    coord = []
    for i in range(node_tags.shape[0]):
        for j in range(node_tags.shape[1]):
            t = node_tags[i, j]
            xyz = gmsh.model.mesh.getNode(t)[0]
            xy = xyz[:-1]
            coord.append(xy)

    coord = np.asarray(coord).reshape(-1, 6)
    #print(coord)# 3d
    #arr_coord = np.concatenate((np.asarray(elem_tags).reshape(-1, 1), coord), axis=1)

    return dict(zip(elem_tags, coord)) #arr_coord'''

def elem_node_coord(element_type: int):
    """An element-tag to node coords dict.
    :param element_type: The type of elements to store in the dict.
    :param tag: The tag of elements to get.
    """
    # msh.nodeTags_elements)
    element_tags, node_tags = gmsh.model.mesh.getElementsByType(2)
    node_tags = np.array_split(node_tags, len(node_tags) / 3)

    coords: List[List[np.ndarray]] = list()
    # msh.node aber nur 2D also muss :-1 weg
    for tags in node_tags:
        coords.append([gmsh.model.mesh.getNode(t)[0][:-1] for t in tags])

    return dict(zip(element_tags, coords))


