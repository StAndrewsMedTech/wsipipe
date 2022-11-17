from pathlib import Path
from typing import List

from PIL.Image import Image
from matplotlib import cm

from pylibCZIrw import czi as pyczi

import numpy as np

from wsipipe.load.slides.slide import SlideBase
from wsipipe.load.slides.region import Region
from wsipipe.utils import Size, Point

def series(int):
    return 1/(2**int)

class CZISlide(SlideBase):
    """
    Read slides to generic format using the pylibCZIrw package.
    For example, to open OMETiff WSIs.
    """
    
    def __init__(self, path: Path) -> None:
        self._path = path
        self._osr = None
        self.wsi = self._getpylibCZIrwSlide()
        self.offset = self.wsi.total_bounding_rectangle[0], self.wsi.total_bounding_rectangle[1]

    def open(self) -> None:
        self._osr = pyczi.CziReader(self.path)

    def close(self) -> None:
        self._osr.close()

    @property
    def path(self) -> Path:
        return self._path
        
    def _getpylibCZIrwSlide(self):
        return pyczi.CziReader(self.path)

    @property
    def dimensions(self) -> List[Size]:
        """ Gets slide dimensions in pixels for all levels in pyramid

        Returns:
            sizelist (List[Size]): A list of sizes
        """

        # Iterate through each "series" and note the x and y dims for each inside level_dims
        total_rect = self.wsi.total_bounding_rectangle
        level_dims={}

        for x in range(10):
            dimensions = int(total_rect[2]*series(x)),int(total_rect[3]*series(x))
            level_dims[x] = dimensions
        return level_dims

    def read_region(self, region: Region) -> Image:
        """Read a region from a WSI

        Loads a specified region from a WSI at a given level

        Args:
            region (Region): A region of the image
        Returns:
            image (Image): A PIL Image of the specified region 
        """
        
        _XYWH =(self.offset[0]+region.location[0],                          #x
                self.offset[1]+region.location[1],                          #y
                int(self.dimensions[region.level][0]/series(region.level)), #w
                int(self.dimensions[region.level][1]/series(region.level))) #h
        region_out = (self.wsi.read(roi=_XYWH, zoom=series(region.level)))
        return region_out

    def read_regions(self, regions: List[Region]) -> List[Image]:
        # TODO: this call could be parallelised
        # though pytorch loaders will do this for us
        regions = [self.read_region(region) for region in regions]
        return regions


