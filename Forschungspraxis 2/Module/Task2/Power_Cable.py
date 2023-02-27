# coding=utf-8
"""
Implementation of the class PowerCable that represents a power cable with 3 conductors inside.
"""

from dataclasses import dataclass
import numpy as np
from pyrit.geometry import Geometry, Circle, Surface
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.constants import mu_0, epsilon_0
import gmsh
from pyrit.geometry import Geometry
from pyrit.bdrycond import BCDirichlet, BdryCond
from pyrit.material import Mat, Materials, Reluctivity
from pyrit.material import Materials, Mat, Permittivity, Conductivity, Reluctivity
from pyrit.shapefunction import TriCartesianEdgeShapeFunction
from pyrit.excitation import Excitations, CurrentDensity
from pyrit.problem import MagneticProblemCartStatic, ElectricProblemCartStatic
from pyrit.toolbox.PostprocessToolbox import plot_field_on_line, get_field_on_line
show_plot = True

@dataclass
class PowerCable:

    '''def __init__(self, k):
        self.k = k'''

    k = 0
    # given
    wire_radius: float = 1.1e-3
    radius_wires_center_points: float = 1.5e-3
    outer_conductor_inner_radius: float = 3e-3
    outer_conductor_outer_radius: float = 3.2e-3
    conductivity_copper: float = 57.7e6
    conductivity_surrounding: float = 0
    relative_permittivity_surrounding: float = 10

    # Strom?
    current: float = 16
    model_name: str = "PowerCable_ys"
    depth: float = 1

    def __post_init__(self):
        self.id_wire_u = 1
        self.id_wire_v = 2
        self.id_wire_w = 3
        self.id_insulation = 10
        self.id_outer_conductor = 11
        self.id_outer_bound = 12

    @property
    def ids_wires(self):
        return [self.id_wire_u, self.id_wire_v, self.id_wire_w]

    @property
    def current_density(self):
        """Current density in [A/m^2]"""
        return self.current / (np.pi * self.wire_radius ** 2)

    def create_problem(self, **kwargs):
        geo = Geometry("Power cable", **kwargs)

        # fragen: Reluktanz?
        materials = Materials(
            mat_wire_u := Mat("Wire u", Permittivity(epsilon_0),  Reluctivity(1 / mu_0),
                              Conductivity(self.conductivity_copper)),
            mat_wire_v := Mat("Wire v", Permittivity(epsilon_0), Reluctivity(1 / mu_0),
                              Conductivity(self.conductivity_copper)),
            mat_wire_w := Mat("Wire w", Permittivity(epsilon_0), Reluctivity(1 / mu_0),
                              Conductivity(self.conductivity_copper)),
            mat_outer_cond := Mat("Outer conductor", Permittivity(epsilon_0), Reluctivity(1 / mu_0),
                                  Conductivity(self.conductivity_copper)),
            mat_insulation := Mat("Insulation", Permittivity(self.relative_permittivity_surrounding * epsilon_0),
                                  Reluctivity(1 / mu_0),
                                  Conductivity(self.conductivity_surrounding)))

        # creating boundary condition
        boundary_conditions = BdryCond(bc := BCDirichlet(0))

        # Creating the excitation: das hier richtig?:
        # setze eine anregung und gebe die jedem wire? in machine slot anders?

        excitations = Excitations(exci := CurrentDensity(self.current_density))

        # given
        pg_wire_u = geo.create_physical_group(self.id_wire_u, 2, "Wire u")
        pg_wire_v = geo.create_physical_group(self.id_wire_v, 2, "Wire v")
        pg_wire_w = geo.create_physical_group(self.id_wire_w, 2, "Wire w")
        pg_insulation = geo.create_physical_group(self.id_insulation, 2, "Insulation")
        pg_outer_conductor = geo.create_physical_group(self.id_outer_conductor, 2, "Outer conductor")
        pg_outer_bound = geo.create_physical_group(self.id_outer_bound, 1, "Outer bound")

        # add material
        geo.add_material_to_physical_group(mat_insulation, pg_insulation)
        geo.add_material_to_physical_group(mat_wire_u, pg_wire_u)
        geo.add_material_to_physical_group(mat_wire_v, pg_wire_v)
        geo.add_material_to_physical_group(mat_wire_w, pg_wire_w)
        geo.add_material_to_physical_group(mat_outer_cond, pg_outer_conductor)

        # add excitation
        if self.k == 1:
            geo.add_excitation_to_physical_group(exci, pg_wire_u)
        elif self.k == 2:
            geo.add_excitation_to_physical_group(exci, pg_wire_v)
        elif self.k == 3:
            geo.add_excitation_to_physical_group(exci, pg_wire_w)
        else:
            geo.add_excitation_to_physical_group(exci, pg_wire_u)
            geo.add_excitation_to_physical_group(exci, pg_wire_v)
            geo.add_excitation_to_physical_group(exci, pg_wire_w)

        # set bc
        geo.add_boundary_condition_to_physical_group(bc, pg_outer_bound)

        # given
        # %% Building model in gmsh and creating the mesh
        with geo:
            # creating the wires?
            circle_u = Circle(self.radius_wires_center_points, 0, 0, self.wire_radius)
            circle_v = Circle(-0.5 * self.radius_wires_center_points, np.sqrt(3) / 2 * self.radius_wires_center_points,
                              0, self.wire_radius)
            circle_w = Circle(-0.5 * self.radius_wires_center_points,
                              -1 * np.sqrt(3) / 2 * self.radius_wires_center_points, 0, self.wire_radius)

            circle_outer_conductor_inner = Circle(0, 0, 0, self.outer_conductor_inner_radius)
            circle_outer_conductor_outer = Circle(0, 0, 0, self.outer_conductor_outer_radius)

            outer_conductor = Surface([circle_outer_conductor_outer], [circle_outer_conductor_inner])

            # insulation ist alles bis auf outer conductor inner und die wires
            insulation = Surface([circle_outer_conductor_inner], [circle_u, ], [circle_v, ], [circle_w, ])
            surface_u = Surface([circle_u])
            surface_v = Surface([circle_v])
            surface_w = Surface([circle_w])

            for surf in (outer_conductor, insulation, surface_u, surface_v, surface_w):
                surf.add_to_gmsh()

            pg_wire_u.add_entity(surface_u)
            pg_wire_v.add_entity(surface_v)
            pg_wire_w.add_entity(surface_w)
            pg_insulation.add_entity(insulation)
            pg_outer_conductor.add_entity(outer_conductor)
            pg_outer_bound.add_entity(circle_outer_conductor_outer)

            geo.create_mesh(2)
            mesh = geo.get_mesh(2)
            regions = geo.get_regions()

        # Defining the shape function for solving the FE problem
        shape_function = TriCartesianEdgeShapeFunction(mesh, self.depth)

        # Setting up the FE problem
        prb_magn = MagneticProblemCartStatic(self.model_name, shape_function, mesh, regions, materials,
                                        boundary_conditions, excitations)

        # Error: AttributeError: 'TriCartesianEdgeShapeFunction' object has no attribute 'elem2regi'
        '''prb_elec = ElectricProblemCartStatic(self.model_name, shape_function, mesh, regions, materials,
                                             boundary_conditions, excitations)'''


        return prb_magn, shape_function


def main():

    # Fragen:
    # Reluktanz?
    # L: kann ja nicht richtig sein: brauche mehrere: Mehrmals simulieren, aber wie?
    ''' Strom nur in einem Wire auf current setzen, anderen 0, dann 3 mal rechnen?
    auch dann habe ich nur 3 Werte: Wie Gegeninduktivität?
    '''
    # für Ti, Tu brauche ich ja L,C

    # keine Möglichkeit von L auf C?
    # C: Genauso nur für elektrostatik?: W= 0.5 * C U^2? nach C umstellen, aber wie U?
    #
    # Vergleich mit machine slot: Durchgehen, ob richtig gemacht in Power_Cable?
    power_cable = PowerCable()
    problem, shape_function = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False)
    load = shape_function.load_vector(problem.regions, problem.excitations)
    X = load / power_cable.current

    # ValueError: Matrix A is singular, because it contains empty row(s)
    solution = problem.solve()
    mesh = problem.mesh
    curlcurl_matrix = solution.curlcurl_matrix


    print('Energy Density', solution.energy_density)
    print('Energy', solution.energy)
    print('Inductivity', 2 * solution.energy / power_cable.current ** 2)
    print('Induct 2', (X.T @ solution.vector_potential / power_cable.current))
    if show_plot:
        solution.plot_energy_density()
        solution.plot_vector_potential()
        solution.plot_current_density()
        plt.show()


    # Compute the magnetic flux density magnitude
    b_abs = np.linalg.norm(solution.b_field, axis=1)

    # Plots the magnetic flux density, style options are 'arrows', 'abs', 'stream'.
    if show_plot:
        solution.plot_equilines()
        solution.plot_b_field('abs')
        solution.plot_b_field('arrows')
        solution.plot_b_field('stream')
        plt.show()

    # Plot a field a long an arbitrary line, with/without smoothing option
    start_point = [0, 0]  # x,y
    end_point = [power_cable.outer_conductor_outer_radius, 0.]  # x,y

    fig, ax = plot_field_on_line(mesh, start_point, end_point, b_abs, lower_bound=0, upper_bound=end_point[0],
                                 smoothing=True, regions=problem.regions, marker='o', label='With smoothing')
    fig, ax = plot_field_on_line(mesh, start_point, end_point, b_abs, lower_bound=0, upper_bound=end_point[0],
                                 smoothing=False, regions=problem.regions, marker='o', fig=fig, ax=ax,
                                 label='Without smoothing')
    ax.legend()
    ax.set_title('Absolute magnetic flux density over the radius')
    ax.set_xlabel('Radius in m')
    ax.set_ylabel(r'$\left|\vec{B}\right|$')
    plt.show()

    # Compare analytic and numerical solution
    radius = np.linspace(1e-5, power_cable.outer_conductor_outer_radius, 200)
    _, ax = plot_field_on_line(mesh, start_point, end_point, solution.vector_potential_field_values, lower_bound=0,
                               upper_bound=end_point[0], smoothing=True, regions=problem.regions, label='Numeric')
    ax.legend()
    ax.set_title("Vector potential")
    plt.show()

    # einzeln die Wires auf Strom setzen?
    '''k = [1, 2, 3]
    L = []
    for i in k:
    # Vergleich mit machine slot: Durchgehen, ob richtig gemacht in Power_Cable?
        power_cable = PowerCable(i)
        problem = power_cable.create_problem(mesh_size_factor=0.2, show_gui=False)

        # ValueError: Matrix A is singular, because it contains empty row(s) -> without Reluktanz
        solution = problem.solve()
        mesh = problem.mesh
        curlcurl_matrix = solution.curlcurl_matrix


        print('Energy Density', solution.energy_density)
        print('Energy', solution.energy)
        L_i = 2 * solution.energy / power_cable.current ** 2
        print('Inductivity', L_i)
        L.append(L_i)

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
            plt.show()'''





if __name__ == '__main__':
    main()
