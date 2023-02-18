import random
from pathlib import Path
from typing import Any, Callable, Optional, Union, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset

class large_data_dist(VisionDataset):
    def __init__(
            self, 
            root: str, 
            transforms: Optional[Callable] = None, 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)