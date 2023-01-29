from dataclasses import dataclass
import numpy as np
import numpy.linalg as la
from typing import Tuple

Point2D = Tuple[float, float]
"""A point in 2D space."""

Point3D = Tuple[float, float, float]
"""A point in 3D space."""


@dataclass
class ShapeFunction:
    """A 2 dimensional shape function."""

    a: float
    b: float
    c: float
    S: float

    def __call__(self, x: float, y: float) -> float:
        """Evaluates this shape function at the point (x,y)."""
        return (self.a + self.b * x + self.c * y) / (2 * self.S)

    @staticmethod
    def of_points(p_i: Point2D, p_j: Point2D, p_k: Point2D):
        """Creates the shape function at the triangle with corner points p_i, p_j and p_k"""

        if len(p_i) != 2 or len(p_j) != 2 or len(p_k) != 2:
            raise Exception("Points must be 2 dimensional!")
        return ShapeFunction.of_coords(p_i[0], p_i[1], p_j[0], p_j[1], p_k[0], p_k[1])

    @staticmethod
    def of_coords(x_i: float, y_i: float, x_j: float, y_j: float, x_k: float, y_k: float):
        """Creates the shape function at the triangle (x_i, y_i), (x_j, y_j), (x_k, y_k)."""

        return ShapeFunction(
            a=x_j * y_k - x_k * y_j,
            b=y_j - y_k,
            c=x_k - x_j,
            S=abs((x_j-x_i)*(y_k-y_i) - (y_j-y_i)*(x_k-x_i)) / 2
        )

    @staticmethod
    def area(p_i: Point2D, p_j: Point2D, p_k: Point2D):
        ax = p_j[0] - p_i[0]
        ay = p_j[1] - p_i[1]
        bx = p_k[0] - p_i[0]
        by = p_k[1] - p_i[1]
        return 0.5*abs(ax*by - ay*bx)

    @staticmethod
    def func_abcS(p_i: Point2D, p_j: Point2D, p_k: Point2D):

        x_i = p_i[0]
        y_i = p_i[1]
        x_j = p_j[0]
        y_j = p_j[1]
        x_k = p_k[0]
        y_k = p_k[1]

        a = x_j * y_k - x_k * y_j
        b = y_j - y_k
        c = x_k - x_j
        S = abs((x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)) / 2
        return a, b, c, S

