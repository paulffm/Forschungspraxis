"""
Functions to generate a current distribution on a HexMesh.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from pyfit.mesh.HexMesh import HexMesh


def current_distribution_simple(mesh: HexMesh) -> csr_matrix:
    """
    Generate a simple current distribution where only 2 edges are excited.

    Parameters
    ----------
    mesh : Mesh.

    Returns
    -------
    csr_matrix with current for each edge.
    """
    idx_x = (np.floor(mesh.num_nodes_x)/3).astype(np.int32) + np.array([1, 2])
    idx_y = (np.floor(mesh.num_nodes_y)/3).astype(np.int32) + np.array([2, 2])
    idx_z = (np.floor(mesh.num_nodes_z)/3).astype(np.int32) + np.array([1, 1])
    idx_source = mesh.ijkc2idx(idx_x, idx_y, idx_z, np.array([1, 1]))
    # idx_source = mesh.ijkc2idx(np.array([1, 2]), np.array([2, 2]), np.array([2, 2]), np.array([1, 1]))
    return coo_matrix((np.ones_like(idx_source), (idx_source, np.zeros_like(idx_source))),
                      shape=(3 * mesh.num_nodes, 1)).tocsr()


def current_distribution_loop(mesh: HexMesh) -> csr_matrix:
    """
    Create a loop current. The current flows in the xy-plane.

    Parameters
    ----------
    mesh : Mesh.

    Returns
    -------
    csr_matrix with current for each edge.
    """
    idx_x1, idx_x2 = np.round(mesh.num_nodes_x / 3).astype(int), np.round(2 * mesh.num_nodes_x / 3).astype(int)
    idx_y1, idx_y2 = np.round(mesh.num_nodes_y / 3).astype(int), np.round(2 * mesh.num_nodes_y / 3).astype(int)
    idx_z = np.round(mesh.num_nodes_z / 3).astype(int)

    idx_line_1 = mesh.ijkc2idx(np.arange(idx_x1, idx_x2 + 1), idx_y1, idx_z, 1)
    idx_line_2 = mesh.ijkc2idx(idx_x2, np.arange(idx_y1, idx_y2 + 1), idx_z, 2)
    idx_line_3 = mesh.ijkc2idx(np.arange(idx_x1, idx_x2 + 1), idx_y2, idx_z, 1)
    idx_line_4 = mesh.ijkc2idx(idx_x1, np.arange(idx_y1, idx_y2 + 1), idx_z, 2)
    i = np.hstack((idx_line_1, idx_line_2, idx_line_3, idx_line_4))
    j = np.zeros_like(i, dtype=np.int8)
    v = np.hstack((np.ones_like(idx_line_1), np.ones_like(idx_line_2), -np.ones_like(idx_line_3),
                   -np.ones_like(idx_line_4)))
    return coo_matrix((v, (i, j)), shape=(3 * mesh.num_nodes, 1), dtype=np.int8).tocsr()


def current_distribution_random(mesh: HexMesh, fill: float = 0.25) -> csr_matrix:
    """
    Create a random current distribution.

    Parameters
    ----------
    mesh : Mesh
    fill : Ratio of edges with current to number of edges.

    Returns
    -------
    csr_matrix with current for each edge.
    """
    if not 0 < fill <= 1:
        raise ValueError("fill must be in (0, 1].")

    num_edges_current = np.round(3 * mesh.num_nodes * fill).astype(np.int64)
    random_generator = np.random.default_rng()
    i = random_generator.integers(0, 3 * mesh.num_nodes, size=num_edges_current)
    j = np.zeros_like(i, dtype=np.int8)
    v = np.ones_like(i, dtype=np.int8)
    return coo_matrix((v, (i, j)), shape=(3 * mesh.num_nodes, 1), dtype=np.int8).tocsr()


def current_distribution_axis(mesh: HexMesh, axis: int, offset1: float = 0.5, offset2: float = 0.5) -> csr_matrix:
    """
    Create a current that flows parallel to an axis.

    Parameters
    ----------
    mesh : Mesh.
    axis : Axis. 1 == x-axis, 2 == y-axis, 3 == z-axis
    offset1 : Offset in the first direction.
    offset2 : Offset in the second direction.

    Returns
    -------
    csr_matrix with current for each edge.
    """
    if not (0 < offset1 < 1 and 0 < offset2 < 1):
        raise ValueError("Offset has to in (0, 1).")

    if axis == 1:  # x-axis
        y = np.round(mesh.num_nodes_y * offset1)
        z = np.round(mesh.num_nodes_z * offset2)
        i = mesh.ijkc2idx(np.arange(mesh.num_nodes_x) + 1, y, z, 1)
    elif axis == 2:  # y-axis
        x = np.round(mesh.num_nodes_x * offset1)
        z = np.round(mesh.num_nodes_z * offset2)
        i = mesh.ijkc2idx(x, np.arange(mesh.num_nodes_y) + 1, z, 2)
    elif axis == 3:  # z-axis
        x = np.round(mesh.num_nodes_x * offset1)
        y = np.round(mesh.num_nodes_y * offset2)
        i = mesh.ijkc2idx(x, y, np.arange(mesh.num_nodes_z) + 1, 3)
    else:
        raise ValueError("Axis has to be 1, 2 or 3.")

    j = np.zeros_like(i)
    v = np.ones_like(i)
    return coo_matrix((v, (i, j)), shape=(3 * mesh.num_nodes, 1)).tocsr()

