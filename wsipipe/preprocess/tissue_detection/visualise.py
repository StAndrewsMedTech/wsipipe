import cv2
import numpy as np
from PIL import Image

from wsipipe.load.datasets import Loader
from wsipipe.preprocess.tissue_detection.tissue_detector import TissueDetector


def visualise_tissue_detection_for_slide(
    slide_path: str, loader: Loader, vis_level:int, tissue_detector: TissueDetector
) -> Image:
    with loader.load_slide(slide_path) as slide:
        thumb = slide.get_thumbnail(vis_level)
    tissue_mask = tissue_detector(thumb)
    tissue_mask = np.array(tissue_mask*255, dtype=np.uint8)
    _, contours, _ = cv2.findContours(
        image=tissue_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )
    # draw contours on the original image
    outline_img = cv2.drawContours(
        image=thumb, contours=contours, contourIdx=-1, 
        color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA
    )
    return Image.fromarray(outline_img)