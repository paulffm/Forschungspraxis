import numpy as np
from dataclasses import dataclass
from Meshsim import Mesh
# Definition of shape function N = a+bx+cy

@dataclass
class ShapeFunction_N:
    depth : float
    element_area : np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


