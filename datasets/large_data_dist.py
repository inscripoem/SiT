import random
import json
from pathlib import Path
from typing import Any, Callable, Optional, Union, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset

class large_data_dist(VisionDataset):
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
        self._dist_folder = Path("./dist/large_data")

        self._images = []
        self._labels = []
        
        if not self._dist_folder.exists():
            self._dist_folder.mkdir(parents=True)
        self._index_file = self._dist_folder / 'large_data_index.json'
        self._train_dist_file = self._dist_folder / 'large_data_train.json'
        self._val_dist_file = self._dist_folder / 'large_data_val.json'
        self._test_dist_file = self._dist_folder / 'large_data_test.json'
        index = self.get_index()
        split_index = self.get_split(index, split)
        if is_pretrain:
            ratio = list(ratio.split("_"))
            if len(ratio) != 2:
                raise ValueError("请检查ratio参数")
            ratio = [int(ratio[0]), int(ratio[1])]
            pretrain_dist_folder = self._dist_folder / 'pretrain'
            if not pretrain_dist_folder.exists():
                pretrain_dist_folder.mkdir(parents=True)
            dist_file = pretrain_dist_folder / f'large_data_neg_{ratio[0]}_pos_{ratio[1]}.json'
            dist_index = self.get_pretrain_dist(split_index, dist_file, ratio)
        else:
            if split == 'train':
                try:
                    ratio = int(ratio)
                except:
                    raise ValueError("请检查ratio参数")
                train_dist_folder = self._dist_folder / 'train'
                if not train_dist_folder.exists():
                    train_dist_folder.mkdir(parents=True)
                dist_file = train_dist_folder / f'large_data_{ratio}.json'
                dist_index = self.get_train_dist(split_index, dist_file, ratio)
            else:
                dist_index = split_index
        
        for data_ in dist_index['data']:
            for cls in dist_index['data'][data_]:
                for image_id in dist_index['data'][data_][cls]:
                    self._images.append(self._base_folder / data_ / cls / (image_id + '.jpg'))
                    self._labels.append(cls)

    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[index]).convert("RGB")
        target = 0 if self._labels[index] == 'neg' else 1
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target
        

    def get_index(self):
        if not self._index_file.exists():
            print('--未检测到索引文件，开始建立索引--')
            index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            total_neg_num = 0
            total_pos_num = 0
            for i in range(6):
                data_ = 'data_' + str(i+1)
                data_folder = self._base_folder / data_
                index['data'].setdefault(data_, {'neg': [], 'pos': []})
                index['num'].setdefault(data_, {'neg_num': 0, 'pos_num': 0})
                for cls in ['neg', 'pos']:
                    for image in (data_folder / cls).glob('*.jpg'):
                        index['data'][data_][cls].append(image.stem)
                        index['num'][data_][cls + '_num'] += 1
            
            for data_ in index['data']:
                total_neg_num += index['num'][data_]['neg_num']
                total_pos_num += index['num'][data_]['pos_num']
            index['num']['total_neg_num'] = total_neg_num
            index['num']['total_pos_num'] = total_pos_num

            with open(self._index_file, 'w') as f:
                json.dump(index, f, sort_keys=True, indent=4, separators=(',', ': '))
            print(f'索引建立完毕，共有{total_neg_num}张阴性样本，{total_pos_num}张阳性样本，其中')
        else:
            print('--检测到索引文件，开始读取索引--')
            with open(self._index_file, 'r') as f:
                index = json.load(f)
            print(f'索引读取完毕，共有{index["num"]["total_neg_num"]}张阴性样本，{index["num"]["total_pos_num"]}张阳性样本，其中')
        print(f'样本分布：{list(index["num"].values())[0:-2]}')
        return index
        
    def get_split(self, index, split):
        if not self._test_dist_file.exists():
            print('--未检测到测试集划分文件，开始建立测试集分布--')
            # 根据每个文件夹的样本数，计算每个文件夹的测试集样本数
            test_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            for data_ in index['data']:
                test_index['num'][data_] = {
                    'neg_num' : index['num'][data_]['neg_num'] // 10, 
                    'pos_num' : index['num'][data_]['pos_num'] // 10
                    }
                test_index['data'][data_] = {
                    'neg' : random.sample(index['data'][data_]['neg'], test_index['num'][data_]['neg_num']), 
                    'pos' : random.sample(index['data'][data_]['pos'], test_index['num'][data_]['pos_num'])
                    }
                test_index['num']['total_neg_num'] += test_index['num'][data_]['neg_num']
                test_index['num']['total_pos_num'] += test_index['num'][data_]['pos_num']
                
            # 将测试集样本写入文件
            with open(self._test_dist_file, 'w') as f:
                json.dump(test_index, f, sort_keys=True, indent=4, separators=(',', ': '))
            # 计算训练验证集样本
            trainval_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            for data_ in index['data']:
                trainval_index['num'][data_] = {
                    'neg_num' : index['num'][data_]['neg_num'] - test_index['num'][data_]['neg_num'],
                    'pos_num' : index['num'][data_]['pos_num'] - test_index['num'][data_]['pos_num']
                    }
                trainval_index['data'][data_] = {
                    'neg' : list(set(index['data'][data_]['neg']) - set(test_index['data'][data_]['neg'])), 
                    'pos' : list(set(index['data'][data_]['pos']) - set(test_index['data'][data_]['pos']))
                    }
                trainval_index['num']['total_neg_num'] += trainval_index['num'][data_]['neg_num']
                trainval_index['num']['total_pos_num'] += trainval_index['num'][data_]['pos_num']
            # 计算验证集样本
            val_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            for data_ in trainval_index['data']:
                val_index['num'][data_] = {
                    'neg_num' : trainval_index['num'][data_]['neg_num'] // 9, 
                    'pos_num' : trainval_index['num'][data_]['pos_num'] // 9
                    }
                val_index['data'][data_] = {
                    'neg' : random.sample(trainval_index['data'][data_]['neg'], val_index['num'][data_]['neg_num']), 
                    'pos' : random.sample(trainval_index['data'][data_]['pos'], val_index['num'][data_]['pos_num'])
                    }
                val_index['num']['total_neg_num'] += val_index['num'][data_]['neg_num']
                val_index['num']['total_pos_num'] += val_index['num'][data_]['pos_num']
            # 将验证集样本写入文件
            with open(self._val_dist_file, 'w') as f:
                json.dump(val_index, f, sort_keys=True, indent=4, separators=(',', ': '))
            # 计算训练集样本
            train_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            for data_ in trainval_index['data']:
                train_index['num'][data_] = {
                    'neg_num' : trainval_index['num'][data_]['neg_num'] - val_index['num'][data_]['neg_num'],
                    'pos_num' : trainval_index['num'][data_]['pos_num'] - val_index['num'][data_]['pos_num']
                    }
                train_index['data'][data_] = {
                    'neg' : list(set(trainval_index['data'][data_]['neg']) - set(val_index['data'][data_]['neg'])), 
                    'pos' : list(set(trainval_index['data'][data_]['pos']) - set(val_index['data'][data_]['pos']))
                    }
                train_index['num']['total_neg_num'] += train_index['num'][data_]['neg_num']
                train_index['num']['total_pos_num'] += train_index['num'][data_]['pos_num']
            # 将训练集与验证集样本写入文件
            with open(self._train_dist_file, 'w') as f:
                json.dump(train_index, f, sort_keys=True, indent=4, separators=(',', ': '))
            print(f'测试集划分建立完毕，共有{test_index["num"]["total_neg_num"]}张阴性样本，{test_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(test_index["num"].values())[2:]}')
            print(f'训练集划分建立完毕，共有{train_index["num"]["total_neg_num"]}张阴性样本，{train_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(train_index["num"].values())[2:]}')
            print(f'验证集划分建立完毕，共有{val_index["num"]["total_neg_num"]}张阴性样本，{val_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(val_index["num"].values())[2:]}')
        else:
            print('--检测到测试集划分文件，开始读取测试集分布--')
        
        print(f'本次构建数据集: {split}')
        if split == 'test':
            with open(self._test_dist_file, 'r') as f:
                test_index = json.load(f)
            print(f'测试集划分读取完毕，共有{test_index["num"]["total_neg_num"]}张阴性样本，{test_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(test_index["num"].values())[0:-2]}')
            return test_index
        elif split == 'train':
            with open(self._train_dist_file, 'r') as f:
                train_index = json.load(f)
            print(f'训练集划分读取完毕，共有{train_index["num"]["total_neg_num"]}张阴性样本，{train_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(train_index["num"].values())[0:-2]}')
            return train_index
        elif split == 'val':
            with open(self._val_dist_file, 'r') as f:
                val_index = json.load(f)
            print(f'验证集划分读取完毕，共有{val_index["num"]["total_neg_num"]}张阴性样本，{val_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(val_index["num"].values())[0:-2]}')
            return val_index

    def get_pretrain_dist(self, split_index, dist_file, ratio):
        if not dist_file.exists():
            print('--未检测到预训练分布文件，开始建立预训练分布--')
            # 根据每个文件夹的样本数，计算每个文件夹的预训练样本数
            dist_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            split_neg_list = []
            split_pos_list = []
            for data_ in split_index['data']:
                split_neg_list.append(split_index['num'][data_]['neg_num'])
                split_pos_list.append(split_index['num'][data_]['pos_num'])
            dist_neg_list = [ratio[0] * i // sum(split_neg_list) for i in split_neg_list]
            if sum(dist_neg_list) == 0:
                if ratio[0] // 6 == 0:
                    dist_neg_list[0] = ratio[0]
                else:
                    dist_neg_list = [ratio[0] // 6 for i in split_neg_list]
            if sum(dist_neg_list) < ratio[0]:
                if ratio[0] - sum(dist_neg_list) > 6:
                    for i in range(6):
                        dist_neg_list[i] += (ratio[0] - sum(dist_neg_list)) // 6
                dist_neg_list[0] += ratio[0] - sum(dist_neg_list)
            dist_pos_list = [ratio[1] * i // sum(split_pos_list) for i in split_pos_list]
            if sum(dist_pos_list) == 0:
                if ratio[1] // 6 == 0:
                    dist_pos_list[0] = ratio[1]
                else:
                    dist_pos_list = [ratio[1] // 6 for i in split_pos_list]
            if sum(dist_pos_list) < ratio[1]:
                if ratio[1] - sum(dist_pos_list) > 6:
                    for i in range(6):
                        dist_pos_list[i] += (ratio[1] - sum(dist_pos_list)) // 6
                dist_pos_list[0] += ratio[1] - sum(dist_pos_list)
            # 随机取出预训练样本
            for i in range(6):
                data_ = list(split_index['data'])[i]
                dist_index['data'][data_] = {
                    'neg': random.sample(split_index['data'][data_]['neg'], dist_neg_list[i]),
                    'pos': random.sample(split_index['data'][data_]['pos'], dist_pos_list[i])
                }
                dist_index['num'][data_] = {
                    'neg_num': dist_neg_list[i],
                    'pos_num': dist_pos_list[i]
                }
                dist_index['num']['total_neg_num'] += dist_neg_list[i]
                dist_index['num']['total_pos_num'] += dist_pos_list[i]
            # 将预训练样本写入文件
            with open(dist_file, 'w') as f:
                json.dump(dist_index, f, sort_keys=True, indent=4, separators=(',', ': '))
            print(f'预训练分布建立完毕，共有{dist_index["num"]["total_neg_num"]}张阴性样本，{dist_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(dist_index["num"].values())[2:]}')
        else:
            print('--检测到预训练分布文件，开始读取预训练分布--')
        with open(dist_file, 'r') as f:
            dist_index = json.load(f)
        print(f'预训练分布读取完毕，共有{dist_index["num"]["total_neg_num"]}张阴性样本，{dist_index["num"]["total_pos_num"]}张阳性样本，其中')
        print(f'样本分布：{list(dist_index["num"].values())[0:-2]}')
        return dist_index
    
    def get_train_dist(self, split_index, dist_file, ratio):
        if not dist_file.exists():
            print('--未检测到下游训练分布文件，开始建立下游训练分布--')
            # 根据每个文件夹的样本数，计算每个文件夹的下游训练样本数
            dist_index: dict = {'num': {'total_neg_num': 0, 'total_pos_num': 0}, 'data': {}}
            split_neg_list = []
            split_pos_list = []
            for data_ in split_index['data']:
                split_neg_list.append(split_index['num'][data_]['neg_num'])
                split_pos_list.append(split_index['num'][data_]['pos_num'])
            dist_neg_list = [i * ratio // 100 for i in split_neg_list]
            dist_pos_list = [i * ratio // 100 for i in split_pos_list]
            # 随机取出下游训练样本
            for i in range(6):
                data_ = list(split_index['data'])[i]
                dist_index['data'][data_] = {
                    'neg': random.sample(split_index['data'][data_]['neg'], dist_neg_list[i]),
                    'pos': random.sample(split_index['data'][data_]['pos'], dist_pos_list[i])
                }
                dist_index['num'][data_] = {
                    'neg_num': dist_neg_list[i],
                    'pos_num': dist_pos_list[i]
                }
                dist_index['num']['total_neg_num'] += dist_neg_list[i]
                dist_index['num']['total_pos_num'] += dist_pos_list[i]
            # 将下游训练样本写入文件
            with open(dist_file, 'w') as f:
                json.dump(dist_index, f, sort_keys=True, indent=4, separators=(',', ': '))
            print(f'下游训练{ratio}%分布建立完毕，共有{dist_index["num"]["total_neg_num"]}张阴性样本，{dist_index["num"]["total_pos_num"]}张阳性样本，其中')
            print(f'样本分布：{list(dist_index["num"].values())[0:-2]}')
        else:
            print('--检测到下游训练分布文件，开始读取下游训练分布--')
        with open(dist_file, 'r') as f:
            dist_index = json.load(f)
        print(f'下游训练{ratio}%分布读取完毕，共有{dist_index["num"]["total_neg_num"]}张阴性样本，{dist_index["num"]["total_pos_num"]}张阳性样本，其中')
        print(f'样本分布：{list(dist_index["num"].values())[0:-2]}')
        return dist_index

    