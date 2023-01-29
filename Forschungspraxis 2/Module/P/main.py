import gmsh
import numpy as np
from matplotlib import pyplot as plt
from shape_function import ShapeFunction
from geo import plot_geometry, elem_node_coord
import sys
from Meshclass import Mesh
#msh = gmsh.model.mesh



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

def main():

    msh = Mesh.create()
    # node_tags arr, node_data arr(koord), element types, element tags, arr 29-200, node_tags_elements, arr länge 516 und bis 101
    #print(msh.num_elements, msh.num_node): 172, 101 weniger elements?
    pg = gmsh.model.getPhysicalGroups()
    # (dim, tag)
    # pg [(1, 3), (2, 1), (2, 2)]

    # Physical Groups

    # In every entry in physical_groups we define the following structure (TAG (dimension, name, indices of all nodes))
    # getNodesforPhysicalGroup(dim, tag)
    physical_groups = dict()
    physical_groups2 = dict()

    for group in pg:
        # physical_groups unterscheidung durch tags: zugriff: physical_groups[3]
        # getphysicalname(dim, tag) -> name as string 'GND'
        # getNodesforPhysicalGroup(dim, tag) -> node_tags, node_coord hier nur tags wegen [0];  # -1 weil man in
        # python ab 0 zählt?

        physical_groups[group[1]] = (group[0], gmsh.model.getPhysicalName(group[0], group[1]),
                                      gmsh.model.mesh.getNodesForPhysicalGroup(group[0], group[1])[0],
                                      gmsh.model.mesh.getNodesForPhysicalGroup(group[0], group[1])[1])
        # so war es vorgegeben: für entity in physical group
        physical_groups2[group[1]] = (group[0], gmsh.model.getPhysicalName(group[0], group[1]),
                                      gmsh.model.mesh.getNodesForPhysicalGroup(group[0], group[1])[0]-1)

        # Dict: (Tag, (Dim, physicalName, array with node tags, array with coord of nodes)
        # 3 GND 1 WIRE 2 SHELL
        # access by tag physical_groups[1] oder [2] [3] dann ein ganzes dict
        # physical_groups[1][2] nur tags von wire
        # physical_groups[2][2] name von wire

    #print(physical_groups[1][2])

    # plot_geometry(physical_groups)
    elem_coord = elem_node_coord(element_type=2)



    # calculate element areas and coefficents a,b, c, S
    elem_areas = [ShapeFunction.area(x[0], x[1], x[2]) for x in elem_coord.values()]
    elem_coeff_p = [ShapeFunction.of_points(x[0], x[1], x[2]) for x in elem_coord.values()]
    print(elem_areas)
    print(elem_coeff_p)
    elem_coeff = np.asarray([ShapeFunction.func_abcS(x[0], x[1], x[2]) for x in elem_coord.values()]).reshape(-1, 4)
    print(elem_coeff)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()



'''  gives me for every physical group die indices der dreiecke/elemente/entities; GND hat keine: nur linie
    elements_shell = (entity_in_physical_group(physical_groups, msh.elem_to_node, 'SHELL')) #2 dim=2 64-171
    elements_wire = (entity_in_physical_group(physical_groups, msh.elem_to_node, 'WIRE')) #1 dim=2 0-63



    node_tags_tri_wire = msh.elem_to_node[elements_wire]
    node_tags_tri_shell = msh.elem_to_node[elements_shell]

    wire = 1
    shell = 2
    gnd = 3

    #plot_geometry(wire, shell, gnd)

    # node_tag, node1, _ = gmsh.model.mesh.getNodes() 303 msh.node 101 x 2 -> bei gmsh werden koord in 3D als lis'''