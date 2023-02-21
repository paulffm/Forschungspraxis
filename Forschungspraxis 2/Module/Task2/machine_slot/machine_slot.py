# coding=utf-8
"""Import the machine slot to a pyrit problem and one example simulation.

.. sectionauthor:: Bundschuh
"""

from typing import List
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0

from pyrit.geometry import Geometry
from pyrit.material import Materials, Mat, Permittivity, Conductivity, Reluctivity
from pyrit.bdrycond import BdryCond, BCDirichlet
from pyrit.excitation import CurrentDensity
from pyrit.problem import Problem
from pyrit.excitation import Excitations, Exci
from pyrit.mesh import TriMesh
from pyrit.shapefunction import TriCartesianEdgeShapeFunction


def create_machine_slot_problem(excitations_left: List[Exci], excitations_right: List[Exci], **kwargs):
    """Create a general problem containing the machine slot model.

    Physical groups are created for each single conductor, their boundaries, the surrounding of the conductors, the
    separator and the outer boundary.

    Parameters
    ----------
    excitations_left : List[Exci]
        A list of excitations. Each excitation in this list is set to one conductor on the left side in the slot
    excitations_right : List[Exci]
        A list of excitations. Each excitation in this list is set to one conductor on the left side in the slot
    kwargs :
        Arguments passed to the constructor of Geometry.

    Returns
    -------
    problem : Problem
        A general problem containing the machine slot model.
    """
    stp_file_name = "Model_Forschungspraxis_CB.stp"
    number_wires = 18  # There are actually 36 single wires because each wire goes through twice (left and right)

    geo = Geometry("machine slot", **kwargs)

    materials = Materials(mat_surrounding_right := Mat(Permittivity(epsilon_0), Reluctivity(1 / mu_0), Conductivity(0)),
                          mat_surrounding_left := Mat(Permittivity(epsilon_0), Reluctivity(1 / mu_0), Conductivity(0)),
                          mat_separator := Mat(Permittivity(epsilon_0), Reluctivity(1 / mu_0), Conductivity(0)),
                          copper := Mat(Permittivity(epsilon_0), Reluctivity(1 / mu_0), Conductivity(1e6)))

    boundary_conditions = BdryCond(bc_outer := BCDirichlet(0))

    excitations = Excitations(*(excitations_left + excitations_right))

    pg_outer_bound = geo.create_physical_group(1, 1, "outer_bound")

    max_tag = 1
    # Physical groups of the boundaries of the right conductors
    pgs_conductor_bound_right = [geo.create_physical_group(max_tag + 1 + k, 1, f"conductor_bound_right_{k}") for k in
                                 range(number_wires)]

    max_tag = pgs_conductor_bound_right[-1].tag
    # Physical groups of the boundaries of the left conductors
    pgs_conductor_bound_left = [geo.create_physical_group(max_tag + 1 + k, 1, f"conductor_bound_left_{k}") for k
                                in range(number_wires)]

    max_tag = pgs_conductor_bound_left[-1].tag
    # Physical groups of the surfaces of the right conductors
    pgs_conductor_surf_right = [geo.create_physical_group(max_tag + 1 + k, 2, f"conductor_surf_right_{k}") for k in
                                range(number_wires)]

    max_tag = pgs_conductor_surf_right[-1].tag
    # Physical groups of the surfaces of the left conductors
    pgs_conductor_surf_left = [geo.create_physical_group(max_tag + 1 + k, 2, f"conductor_surf_left_{k}") for k in
                               range(number_wires)]

    max_tag = pgs_conductor_surf_left[-1].tag
    pg_surrounding_right = geo.create_physical_group(max_tag + 1, 2, "Right surrounding")
    pg_surrounding_left = geo.create_physical_group(max_tag + 2, 2, "Left surrounding")
    pg_separator = geo.create_physical_group(max_tag + 3, 2, "Separator")

    geo.add_material_to_physical_group(mat_surrounding_right, pg_surrounding_right)
    geo.add_material_to_physical_group(mat_surrounding_left, pg_surrounding_left)
    geo.add_material_to_physical_group(mat_separator, pg_separator)
    geo.add_material_to_physical_group(copper, *pgs_conductor_surf_left)
    geo.add_material_to_physical_group(copper, *pgs_conductor_surf_right)

    geo.add_boundary_condition_to_physical_group(bc_outer, pg_outer_bound)

    for exci, pg in zip(excitations_left + excitations_right, pgs_conductor_surf_left + pgs_conductor_surf_right):
        geo.add_excitation_to_physical_group(exci, pg)

    with geo:
        gmsh.merge(stp_file_name)

        gmsh.model.occ.remove([(2, 1), (2, 2)])
        gmsh.model.occ.removeAllDuplicates()

        # Indices of the curves surrounding the conductors at the left and right side
        conductor_bounds_right = [[10 + 2 * k, 11 + 2 * k] for k in range(number_wires)]
        conductor_bounds_left = [[56 + 2 * k, 57 + 2 * k] for k in range(number_wires)]

        # Indices of the wires (gmsh internally) of the conductors at the left and right side
        wires_right, wires_left = [], []
        # Indices of the surfaces of the conductors at the left and right side
        conductor_surfaces_right, conductor_surfaces_left = [], []

        for k in range(number_wires):
            wires_right.append(wire := gmsh.model.occ.addWire(conductor_bounds_right[k]))
            conductor_surfaces_right.append(gmsh.model.occ.addPlaneSurface([wire, ], 100 + k))

            wires_left.append(wire := gmsh.model.occ.addWire(conductor_bounds_left[k]))
            conductor_surfaces_left.append(gmsh.model.occ.addPlaneSurface([wire, ], 118 + k))

        # Indices of the Wires of the left and right surrounding of the conductors and the wire of the separating region
        wire_right = gmsh.model.occ.addWire(np.arange(1, 10))
        wire_left = gmsh.model.occ.addWire(np.arange(46, 56))
        wire_separator = gmsh.model.occ.addWire([93, 9, 8, 92, 48, 47])

        # Indices of the surfaces of the left and right surrounding of the conductors and of the separating region
        surrounding_right = gmsh.model.occ.addPlaneSurface([wire_right] + wires_right)
        surrounding_left = gmsh.model.occ.addPlaneSurface([wire_left] + wires_left)
        separator = gmsh.model.occ.addPlaneSurface([wire_separator])

        gmsh.model.occ.remove([(1, 92), ], recursive=True)

        # Assign entities
        pg_surrounding_right.add_entity(surrounding_right)
        pg_surrounding_left.add_entity(surrounding_left)
        pg_separator.add_entity(separator)
        pg_outer_bound.add_entity(*(np.arange(1, 8).tolist() + [46] + np.arange(49, 56).tolist() + [92, 93]))

        for pg, tag in zip(pgs_conductor_bound_left + pgs_conductor_bound_right,
                           conductor_bounds_left + conductor_bounds_right):
            pg.add_entity(*tag)

        for pg, tag in zip(pgs_conductor_surf_left + pgs_conductor_surf_right,
                           conductor_surfaces_left + conductor_surfaces_right):
            pg.add_entity(tag)

        geo.create_mesh(dim=2)
        mesh = geo.get_mesh(dim=2)
        regions = geo.get_regions()

    mesh.node = mesh.node / 1000  # Rescale because the gmsh model is in mm. Now the mesh is in m `
    problem = Problem("Machine slot", mesh, None, regions, materials, boundary_conditions, excitations)

    return problem


def simulate_mqs():
    """Simulate the slot as a MQS problem."""

    frequency = 50  # The frequency of the problem
    omega = 2 * np.pi * frequency  # Angular frequency

    vals = np.zeros(18)  # Value of the current density of the conductors
    vals[4] = 1
    excis_left = [CurrentDensity(val) for val in vals]  # List of excitations for the left side in the slot
    excis_right = [CurrentDensity(val) for val in vals]  # List of excitations for the right side in the slot

    problem = create_machine_slot_problem(excis_left, excis_right, show_gui=True, mesh_size_factor=0.03)

    mesh: TriMesh = problem.mesh
    shape_function = TriCartesianEdgeShapeFunction(mesh)
    problem.shape_function = shape_function

    # region Build and solve the system

    curlcurl = shape_function.curlcurl_operator(problem.regions, problem.materials, Reluctivity)
    mass = shape_function.mass_matrix(problem.regions, problem.materials, Conductivity)

    matrix = curlcurl + 1j * omega * mass
    load = shape_function.load_vector(problem.regions, problem.excitations)

    matrix_shrink, rhs_shrink, _, _, support_data = shape_function.shrink(matrix, load, problem, 1)
    a_shrink, _ = type(problem).solve_linear_system(matrix_shrink.tocsr(), rhs_shrink.tocsr())
    vector_potential = shape_function.inflate(a_shrink, problem, support_data)

    # endregion

    # region Post-processing

    b_field = shape_function.curl(np.real(vector_potential))

    mesh.plot_scalar_field(np.real(vector_potential))
    mesh.plot_scalar_field(np.linalg.norm(b_field, axis=1), title="Absolute b field")
    mesh.plot_equilines(np.real(vector_potential))

    plt.show()

    # endregion


def main():
    simulate_mqs()


if __name__ == '__main__':
    main()
