from pathlib import Path
from typing import List

from PIL.Image import Image
from openslide import open_slide

from wsipipe.load.slides.slide import SlideBase
from wsipipe.load.slides.region import Region
from wsipipe.utils import Size, Point


class OSSlide(SlideBase):
    """
    Read slides to generic format using the openslide package.
    For example, to open OMETiff WSIs.
    """
    
    def __init__(self, path: Path) -> None:
        self._path = path
        self._osr = None

    def open(self) -> None:
        self._osr = open_slide(str(self._path))

    def close(self) -> None:
        self._osr.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimensions(self) -> List[Size]:
        """ Gets slide dimensions in pixels for all levels in pyramid

        If fewer than 10 levels exist in the pyramid it calculates the 
        extra sizes and adds them to the list

        Returns:
            sizelist (List[Size]): A list of sizes
        """
        # TODO: review limits - make sure to docuement
        sizelist = [Size(*dim) for dim in self._osr.level_dimensions]
        size_smallest_level = sizelist[-1]
        size_smallest_level = min(size_smallest_level.width, size_smallest_level.height)
        nlevels = len(sizelist)
        while nlevels < 10:
            max_level_dim = sizelist[-1]
            next_level_size = Size(int(max_level_dim.width // 2), int(max_level_dim.height // 2))
            sizelist.append(next_level_size)
            size_smallest_level = sizelist[-1]
            size_smallest_level = min(size_smallest_level.width, size_smallest_level.height)
            nlevels = len(sizelist)
        return sizelist

    def check_level(self, region: Region) -> bool:
        """ Checks if level specified in region exists in pyramid
        Args:
            region (Region): A Region to check
        Returns:
            (bool): True if level in region exists in pyramid
        """
        level_count = self._osr.level_count
        return region.level < level_count

    def convert_region(self, region: Region) -> Image:
        """ Creates a PIL image of a region by downsampling from lower level
        Args:
            region (Region): A Region to create
        Returns:
            image (Image): A downsampled PIL Image
        """
        max_level = self._osr.level_count - 1
        level_diff = region.level - max_level
        size_at_max_lev = Size(region.size.width * (2 ** level_diff), region.size.height * (2 ** level_diff))
        x_at_max_lev = region.location.x * (2 ** level_diff)
        y_at_max_lev = region.location.y * (2 ** level_diff)
        new_region = Region(location=Point(x_at_max_lev, y_at_max_lev), size=size_at_max_lev, level=max_level)
        image = self._osr.read_region(new_region.location, new_region.level, new_region.size)
        image = image.resize((region.size))
        return image

    def read_region(self, region: Region) -> Image:
        """Read a region from a WSI

        Checks if the specified level for the region exists in the pyramid.
        If not reads the region from the highest level that exists and downscales it

        Args:
            region (Region): A region of the image
        Returns:
            image (Image): A PIL Image of the specified region 
        """
        if self.check_level(region):
            region_out = self._osr.read_region(region.location, region.level, region.size)
        else:
            region_out = self.convert_region(region)
        return region_out

    def read_regions(self, regions: List[Region]) -> List[Image]:
        # TODO: this call could be parallelised
        # though pytorch loaders will do this for us
        regions = [self.read_region(region) for region in regions]
        return regions


