import numpy as np
from typing import Union, Tuple, NoReturn
from scipy.sparse import csr_matrix, coo_matrix
from scipy.constants import constants
from pyfit.shapes.Shape3D import Shape3D
from pyfit.shapes.Shape2D import Shape2D
from pyfit.shapes.Brick import Brick
from pyfit.shapes.Rectangle import Rectangle
from pyfit.mesh.HexMesh import HexMesh
from pyfit.shapefunctions.EdgeShapeFunctions import EdgeShapeFunctions
from pyfit.solver import TimeDomainSolver
import SignalGeneration


class Box:
    """Class representing a box resonator used in all tasks."""
    def __init__(self, sides_box: np.ndarray, origin_box: np.ndarray = None, permeability: float = constants.mu_0,
                 permittivity: float = constants.epsilon_0) -> None:
        """
        Constructor.

        Parameters
        ----------
        sides_box : np.ndarray
            (3,) array. Sides of the box in [m].
        origin_box : np.ndarray
            (3,) array. x, y, z-coordinates of box's origin.
        """
        if origin_box is None:
            origin_box = - sides_box / 2
        self.shapes = {1: Brick(origin_box, sides_box)}
        self.shape2permeability = {1: permeability}
        self.shape2permittivity = {1: permittivity}
        self._mesh: HexMesh = None

    def add_shape(self, shape: Union[Rectangle, Brick], shape_id: int, permeability_rel: float = 1,
                  permittivity_rel: float = 1) -> NoReturn:
        """
        Add a shape within the box resonator.

        Parameters
        ----------
        shape : Union[Rectangle, Brick]
            2 or 3 dimensional shape.
        shape_id : int
            Id of the shape.
        permeability_rel : float
            Relative permeability. Default value is 1.
        permittivity_rel
            Relative permittivity. Default value is 1.
        """
        if shape_id in self.shapes.keys():
            raise ValueError(f"Id {shape_id} is already used.")
        if permeability_rel <= 0 or permittivity_rel <= 0:
            raise ValueError("Relative permittivity and relative permeability must be greater than zero.")
        if self.has_mesh:
            raise UserWarning("Mesh already exists. Hence, shape could not be added.")
        self.shapes[shape_id] = shape
        self.shape2permeability[shape_id] = constants.mu_0 * permeability_rel
        self.shape2permittivity[shape_id] = constants.epsilon_0 * permittivity_rel

    def idx_dual_edges_nontrivial(self) -> np.ndarray:
        """
        Return the indices of nontrivial dual edges. These indices can be found in the reluctance matrix.

        Returns
        -------
        np.ndarray
            Indices of nontrivial dual edges.
        """
        sft = EdgeShapeFunctions(self.mesh)
        reluctance_matrix = sft.face_mass(np.array([1]) * (self.mesh.elem2regi == 1))
        return reluctance_matrix.indices

    def idx_primary_edges_nontrivial(self) -> np.ndarray:
        """
        Return the indices of nontrivial primary edges.

        Returns
        -------
        np.ndarray
            Indices of nontrivial primary edges.
        """
        return np.where(self.mesh.primary_edge_lengths)[0]

    def electric_boundary_box(self, *args: str) -> np.ndarray:
        """
        Return the nontrivial primary indices after setting electric boundary conditions at given boundaries of the
        box resonator.

        Parameters
        ----------
        args : str
            "front", "back", "top", "bottom", "left", "right"
        Returns
        -------
        idx_primary_edges : np.ndarray
            Nontrivial primary indices after setting electric boundary conditions.
        """
        idx_boundary = self.mesh.idx_boundary("tangential", *args)
        idx_primary_edges = np.setdiff1d(self.idx_primary_edges_nontrivial(), idx_boundary)
        return idx_primary_edges

    def operators(self, idx_primary_edges: np.ndarray, idx_dual_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                                            csr_matrix]:
        """
        Compute the reluctance-, capacitance- matrix and the curl_operator.

        Parameters
        ----------
        idx_primary_edges : np.ndarray
            (n,) array. Indices of nontrivial primary edges after considering boundary conditions.
        idx_dual_edges : np.ndarray
            (m,) array. Indices of nontrivial dual edges after considering boundary conditions.

        Returns
        -------
        reluctance_dg : np.ndarray
            Reluctance matrix (dual grid).
        capacitance_pg : np.ndarray
            Capacitance matrix (primary grid).
        curl_operator : csr_matrix
            Curl operator (dual grid, primary grid).
        """
        reluctance_dg = 1 / self.shape2permeability[1] * self.mesh.dual_edge_lengths[idx_dual_edges] / \
                        self.mesh.primary_face_areas[idx_dual_edges]
        reluctance_dg.shape = (idx_dual_edges.shape[0], 1)
        capacitance_pg = self.shape2permittivity[1] * self.mesh.dual_face_areas[idx_primary_edges] / \
                         self.mesh.primary_edge_lengths[idx_primary_edges]
        capacitance_pg.shape = (idx_primary_edges.shape[0], 1)
        curl_operator = (self.mesh.primary_curl[idx_dual_edges, :])[:, idx_primary_edges]
        return reluctance_dg, capacitance_pg, curl_operator

    def shape2idx(self, shape_id: int, direction: str = None) -> np.ndarray:
        """
        Get the indices belonging to a shape.

        Parameters
        ----------
        shape_id : int
            Id of the shape.
        direction : str
            Direction of the edges. "x", "y", "z". Optional.

        Returns
        -------
        idx : np.ndarray
            Indices of the shape. If a direction is considered only matching indices are returned.
        """
        if isinstance(self.shapes[shape_id], Shape3D):
            idx = np.where(self.mesh.elem2regi == shape_id)[0]
        elif isinstance(self.shapes[shape_id], Shape2D):
            idx = np.where(self.mesh.face2regi == shape_id)[0]
        else:
            raise NotImplementedError

        if direction == "x":
            return idx[idx < self.mesh.num_nodes]
        if direction == "y":
            return idx[(idx >= self.mesh.num_nodes) * (idx < 2 * self.mesh.num_nodes)]
        if direction == "z":
            return idx[idx >= 2 * self.mesh.num_nodes]
        return idx

    @property
    def has_mesh(self):
        """Bool if the mesh has been set."""
        return isinstance(self._mesh, HexMesh)

    @property
    def mesh(self):
        """Property for the mesh. Raises UserWarning if mesh is not set."""
        if self.has_mesh:
            return self._mesh
        raise UserWarning("Mesh is not set!")

    @property
    def idx_primary_edges(self):
        return self.idx_primary_edges_nontrivial()

    def set_mesh(self, mesh_size_max: np.ndarray, verbose: bool = False) -> NoReturn:
        """
        Generate the mesh.

        Parameters
        ----------
        mesh_size_max : np.ndarray
            (3,) array. Maximal mesh sizes in x, y, z-direction.
        verbose : bool
            Flag for plots.
        """
        self._mesh = HexMesh("box_resonator", tuple(self.shapes.values()), tuple(self.shapes.keys()), mesh_size_max,
                             verbose=verbose)

    def get_t_axis_leapfrog(self, signal: SignalGeneration.Signal) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Return the time axi needed for the leapfrog algorithm.

        Parameters
        ----------
        signal : SignalGeneration.Signal
            Object of Signal.Generation.Signal. Contains properties of signal.

        Returns
        -------
        t_axis : Tuple[np.ndarray, np.ndarray, np.ndarray]
            (t_axis_primary, t_axis_dual, t_step)
        """
        t_step = TimeDomainSolver.cfl_condition(self.mesh, self.idx_primary_edges, constants.mu_0,
                                                constants.epsilon_0)
        return TimeDomainSolver.time_axis(signal.t_start, signal.t_end, t_step)


class BoxResonator(Box):
    """Class to represent the box resonator as used in tasks 1 to 3. Electric boundary conditions are assumed on all
    outside surfaces of the box."""

    def __init__(self, sides_box: np.ndarray, origin_box: np.ndarray = None) -> None:
        """
        Constructor for box with electrodes according to tasks 1,2 and 3.

        Parameters
        ----------
        sides_box : np.ndarray
            (3,) array. Sides of the box in [m].
        origin_box : np.ndarray
            (3,) array. Origin of the box. The default is none.
        """
        super().__init__(sides_box, origin_box)

    @property
    def idx_primary_edges(self) -> np.ndarray:
        """Get the indices of all primary edges considering electric boundary conditions."""
        return self.electric_boundary_box("left", "right", "front", "back", "top", "bottom")

    @property
    def idx_dual_edges(self) -> np.ndarray:
        """Get the indices of all dual edges after considering magnetic boundary conditions at the bottom of the box."""
        return self.idx_dual_edges_nontrivial()

    def get_operators(self):
        return super().operators(self.idx_primary_edges, self.idx_dual_edges)


class BoxWithElectrodes(Box):
    """Class to represent the box resonator with two electrodes used in tasks 4 and 5. Electric boundary conditions are
    assumed at the left, right, front, back and top. Magnetic boundary conditions are assumed at the bottom. The
    electrodes are placed on the bottom pane."""

    def __init__(self, sides_box: np.ndarray, sides_electrode: np.ndarray, id_electrode_a: int = 11,
                 id_electrode_b: int = 12, origin_box: np.ndarray = None) -> None:
        """
        Constructor for box with electrodes according to task 4.

        Parameters
        ----------
        sides_box : np.ndarray
            (3,) array. Sides of the box in [m].
        sides_electrode : np.ndarray
            (3,) array. Width of the side of the electrodes.
        id_electrode_a : int
            ID of electrode a. The default is 11.
        id_electrode_b : int
            ID of electrode b. The default is 12.
        origin_box : np.ndarray
            (3,) array. Origin of the box. The default is none.
        """
        super().__init__(sides_box, origin_box)

        # determine origin of electrode a and b according to task4:
        sides_box = self.shapes[1].side
        origin_a = self.shapes[1].origin + np.array([0.06, sides_box[1, 1] / 2, 0])
        origin_b = self.shapes[1].origin + np.array([sides_box[0, 0] - 0.06, sides_box[1, 1] / 2, 0])

        # add electrode a and b
        self.add_shape(Rectangle(origin_a, sides_electrode), id_electrode_a)
        self.add_shape(Rectangle(origin_b, sides_electrode), id_electrode_b)

        self.id_electrode_a, self.id_electrode_b = id_electrode_a, id_electrode_b

    @property
    def idx_primary_edges(self) -> np.ndarray:
        """Get the indices of all primary edges considering electric boundary conditions at the left, right,
        front, back and top of the box."""
        return self.electric_boundary_box("left", "right", "front", "back", "top")

    @property
    def idx_dual_edges(self) -> np.ndarray:
        """Get the indices of all dual edges after considering magnetic boundary conditions at the bottom of the box."""
        return self.idx_dual_edges_nontrivial()

    @property
    def idx_electrode_a_x(self):
        """Get the indices in x-direction for electrode A."""
        return np.intersect1d(self.shape2idx(self.id_electrode_a, "x"), self.idx_primary_edges)

    @property
    def idx_electrode_b_x(self):
        """Get the indices in x-direction for electrode B."""
        return np.intersect1d(self.shape2idx(self.id_electrode_b, "x"), self.idx_primary_edges)

    def current_distribution_electrode(self, id_electrode: int, idx_primary_edges: np.ndarray = np.array([])) \
            -> csr_matrix:
        """
        Create the current distribution operator for an electrode. The current distribution is supposed to be
        homogenous. Hence, value of the current distribution operator on the outside edges is set to 0.5
        (Only half the area described by an outside edge is part of the electrode). A current is applied solely
        in x direction.

        Parameters
        ----------
        id_electrode : int
            Indices of the electrode in x-direction.
        idx_primary_edges : np.ndarray
            Indices of all considered nontrivial primary edges. Optional.

        Returns
        -------
        csr_matrix
            current distribution operator.
        """
        if id_electrode == self.id_electrode_a:
            idx_electrode = self.idx_electrode_a_x
        elif id_electrode == self.id_electrode_b:
            idx_electrode = self.idx_electrode_b_x
        else:
            raise ValueError("No valid ID of an electrode given!")

        if idx_primary_edges.shape[0] == 0:
            idx_primary_edges = self.idx_primary_edges

        step_width = np.diff(idx_electrode)
        num_x_parallel = step_width[step_width > 1].shape[0] + 1
        num_x_line = int(idx_electrode.shape[0] / num_x_parallel)
        v = np.ones_like(idx_electrode, dtype=np.float16)
        v[:num_x_line] = .5
        v[-num_x_line:] = .5
        return coo_matrix((v, (idx_electrode, np.zeros_like(idx_electrode))),
                          shape=(idx_primary_edges.shape[0], 1)).tocsr()

    def cube2edges(self, x: float, y: float, z: float, width: float = .02,
                   abs_tol: float = 1e-12) -> np.ndarray:
        """
        Get tangential edges part of the considered cube.

        Parameters
        ----------
        x : float
            x-position of the cube's lower left corner.
        y : float
            y-position of the cube's lower left corner.
        z : float
            z-position of the cube's lower left corner.
        width : float
            Width of the cube.
        abs_tol : float
            Absolute tolerance for lookup.

        Returns
        -------
        edges_cube : np.ndarray
            Indices of the cubes edges.
        """
        # get x,y,z-indices that are part of the cube:
        x_12 = np.where((self.mesh.x_coord >= x - abs_tol) * (self.mesh.x_coord <= x + width + abs_tol))[0]
        y_12 = np.where((self.mesh.y_coord >= y - abs_tol) * (self.mesh.y_coord <= y + width + abs_tol))[0]
        z_12 = np.where((self.mesh.z_coord >= z - abs_tol) * (self.mesh.z_coord <= z + width + abs_tol))[0]

        # filter for the smallest and greatest x,y,z-indices:
        x1, x2 = np.min(x_12), np.max(x_12)
        y1, y2 = np.min(y_12), np.max(y_12)
        z1, z2 = np.min(z_12), np.max(z_12)

        # compute the indices of the edges by using canonical indexing:
        edges_x = (np.arange(x1, x2)[:, np.newaxis, np.newaxis]
                   + np.arange(y1, y2 + 1)[np.newaxis, :, np.newaxis] * self.mesh.num_nodes_x
                   + np.arange(z1, z2 + 1)[np.newaxis, np.newaxis, :] * self.mesh.num_nodes_xy).flatten()
        edges_y = (np.arange(x1, x2 + 1)[:, np.newaxis, np.newaxis]
                   + np.arange(y1, y2)[np.newaxis, :, np.newaxis] * self.mesh.num_nodes_x
                   + np.arange(z1, z2 + 1)[np.newaxis, np.newaxis, :] * self.mesh.num_nodes_xy
                   ).flatten() + self.mesh.num_nodes
        edges_z = (np.arange(x1, x2 + 1)[:, np.newaxis, np.newaxis]
                   + np.arange(y1, y2 + 1)[np.newaxis, :, np.newaxis] * self.mesh.num_nodes_x
                   + np.arange(z1, z2)[np.newaxis, np.newaxis, :] * self.mesh.num_nodes_xy
                   ).flatten() + 2 * self.mesh.num_nodes

        return np.hstack((edges_x, edges_y, edges_z))

    def boundary_cube(self, x: float, y: float, z: float, width: float = .02):
        """
        Get the indices of the non-trivial primary edges after considering electric boundary conditions on a cube.

        Parameters
        ----------
        x : float
            x-position of the cube's lower left corner.
        y : float
            y-position of the cube's lower left corner.
        z : float
            z-position of the cube's lower left corner.
        width : float
            Width of the cube.

        Returns
        -------
        idx_pe : np.ndarray
            Indieces of primary edges.
        """
        # get idx of tangential edges of cube -> electric boundary conditions:
        idx_cube = self.cube2edges(x, y, z, width=width)
        return np.setdiff1d(self.idx_primary_edges, idx_cube)

    def get_properties(self):
        properties = {
            "box_sides": self.shapes[1].side,
            "box_origin": self.shapes[1].origin,
            "box_permeability": self.shape2permeability[1],
            "box_permittivity": self.shape2permittivity[1],
            "electrode_a_size": self.shapes[self.id_electrode_a].side,
            "electrode_a_origin": self.shapes[self.id_electrode_a].origin,
            "electrode_b_size": self.shapes[self.id_electrode_b].side,
            "electrode_b_origin": self.shapes[self.id_electrode_b].origin,
        }
        if self.mesh is not None:
            properties["num_mesh_nodes"] = self.mesh.num_nodes

        return properties

