from typing import NamedTuple


class Point(NamedTuple):
    """an x y point in integers"""
    x: int
    y: int


class PointF(NamedTuple):
    """an x y point in floating numbers"""
    x: float
    y: float


class Address(NamedTuple):
    """a row and column point"""
    row: int
    col: int


class Size(NamedTuple):
    """size given by width and height"""
    width: int
    height: int

    def as_shape(self):
        return Shape(self.height, self.width)


class Shape(NamedTuple):
    """chape given by rows and columns"""
    num_rows: int
    num_cols: int

    def as_size(self):
        return Size(self.num_cols, self.num_rows)