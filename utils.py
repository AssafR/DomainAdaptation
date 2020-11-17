# Source: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
from PIL import Image
from enum import Enum
import numpy as np

def loop_iterable(iterable):
    while True:
        yield from iterable


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class NET_ARCHICECTURE(Enum):
    ONE_FC = 1
    TWO_FC = 2
    THREE_FC = 3
               
        
        
