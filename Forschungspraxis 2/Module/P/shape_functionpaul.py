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
    def of_coords(lst):
        """Creates the shape function at the triangle (x_i, y_i), (x_j, y_j), (x_k, y_k)."""
        x_i = lst[0]
        y_i = lst[1]
        x_j = lst[2]
        y_j = lst[3]
        x_k = lst[4]
        y_k = lst[5]

        return ShapeFunction(
            a=x_j * y_k - x_k * y_j,
            b=y_j - y_k,
            c=x_k - x_j,
            S=abs((x_j-x_i)*(y_k-y_i) - (y_j-y_i)*(x_k-x_i)) / 2
        )

    @staticmethod
    def area(lst): # (p_i: Point2D, p_j: Point2D, p_k: Point2D):
        x_i = lst[0]
        y_i = lst[1]
        x_j = lst[2]
        y_j = lst[3]
        x_k = lst[4]
        y_k = lst[5]

        ax = x_j - x_i
        ay = y_j - y_i
        bx = x_k - x_i
        by = y_k - y_i
        return 0.5*abs(ax*by - ay*bx)

'''        ax = p_j[0] - p_i[0]
        ay = p_j[1] - p_i[1]
        bx = p_k[0] - p_i[0]
        by = p_k[1] - p_i[1]'''