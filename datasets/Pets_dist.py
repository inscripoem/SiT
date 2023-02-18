import random
from pathlib import Path
from typing import Any, Callable, Optional, Union, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset

class pets_dist(VisionDataset):
    def __init__(
        self, 
        root: str, 
        split: str = "trainval",
        is_pretrain: bool = True,
        ratio: str = "",
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = Path(self.root)
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"
        self._dist_folder = Path("./dist/Pets")

        self._image_ids = []
        self._labels = []
        
        if not self._dist_folder.exists():
            self._dist_folder.mkdir(parents=True)
        if is_pretrain:
            ratio = list(ratio.split("_"))
            dist_file = self._dist_folder / f"Pets_dog_{ratio[0]}_cat_{ratio[1]}.txt"
        if not dist_file.exists():
            dog_ids = []
            cat_ids = []
            with open(self._anns_folder / "list.txt") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    image_id, _, label, *_ = line.strip().split()
                    if label == '1':
                        cat_ids.append(image_id)
                    else:
                        dog_ids.append(image_id)
                print(f'Pets数据集读取完毕, 共有{len(dog_ids)}只狗，{len(cat_ids)}只猫')
            if is_pretrain:
                dog_ids = random.sample(dog_ids, int(ratio[0]))
                cat_ids = random.sample(cat_ids, int(ratio[1]))
                self._image_ids = dog_ids + cat_ids
                self._labels = ['dog']*len(dog_ids) + ['cat']*len(cat_ids)
                print(f'Pets数据集划分完毕, 随机抽取{len(dog_ids)}只狗，{len(cat_ids)}只猫')
                with open(dist_file, 'w') as f:
                    for image_id, label in zip(self._image_ids, self._labels):
                        f.write(f'{image_id} {label}\n')
        else:
            with open(dist_file) as file:
                for line in file:
                    image_id, label = line.strip().split()
                    self._image_ids.append(image_id)
                    self._labels.append(label)
            print(f'Pets数据集读取完毕, 共有{len(self._image_ids)}张图片，前两张图片的文件名为{self._image_ids[0]}和{self._image_ids[1]}')
        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in self._image_ids]


        

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in self._image_ids]

    def __len__(self) -> int:
        return len(self._images)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[index]).convert("RGB")
        target: Any = []
        target.append(self._labels[index])
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target