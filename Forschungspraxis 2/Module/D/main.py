import gmsh
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange
from geometry import plot_geometry, cable, element_node_tags, element_node_coords
from shape_function import ShapeFunction
import meshio
import gmsh
from mesh import Mesh



def main():

    cable()
    mesh = Mesh.create()
    res = element_node_coords(2)
    print(res)
    y = res.values()

    #print(y)
    #print(y[0])
    # koordinaten der punkte für ein dreieck
    element_areas = [ShapeFunction.area(x[0], x[1], x[2]) for x in res.values()]
    #for x in res.values():
    #    print(x)



if __name__ == '__main__':
    main()