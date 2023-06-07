import random
import json
from pathlib import Path
from typing import Any, Callable, Optional, Union, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset

class large_general(VisionDataset):
    def __init__(self, root: str, transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)

        self._images = []
        self._labels = []

        subdirs = [x for x in self.root.iterdir() if x.is_dir() and x.name.startswith('data')]

        for subdir in subdirs:
            neg_path = subdir / 'neg'
            pos_path = subdir / 'pos'
            for img in neg_path.iterdir():
                self._images.append(img)
                self._labels.append(0)
            for img in pos_path.iterdir():
                self._images.append(img)
                self._labels.append(1)
        
    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[index]).convert("RGB")
        target = self._labels[index]
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target