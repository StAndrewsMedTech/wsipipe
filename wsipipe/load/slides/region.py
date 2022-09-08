from typing import NamedTuple, Tuple

from wsipipe.utils import Point, Size


class Region(NamedTuple):
    """Class for a Region of a whole slide image
    Args:
        level (int): Level to extract the region
        location (Point): x y tuple giving location of top left of region at that level
        size (Size): width and height tuple giving size of region at that level
    """
    level: int
    location: Point
    size: Size

    @classmethod
    def make(cls, x: int, y: int, size: int, level: int):
        """An alternate construction method for square region
        
        Assumes a square region of width and height equal to size

        Args:
            x (int): the pixel location of left of image at level
            y (int): the pixel location of top of image at level
            size (int): size of square region
            level (int): Level to extract the region
            
        """
        location = Point(x, y)
        size = Size(size, size)
        return Region(level, location, size)

    def as_values(self) -> Tuple[int, int, int, int, int]:
        """Splits out location and size into separate values"""
        return (
            self.location.x,
            self.location.y,
            self.size.width,
            self.size.height,
            self.level,
        )