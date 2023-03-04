import torch
from datasets import datasets_utils
from torch.utils.data import DataLoader
from datasets import Pets_dist, large_data_dist
from torchvision import transforms

class DistSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
    def __iter__(self):
        for _ in range(self.num_batches):
            yield torch.randperm(len(self.dataset)).tolist()[:self.batch_size]
    def __len__(self):
        return int(self.num_batches)


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
dataset = Pets_dist.pets_dist('G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet', split='test', is_pretrain=False, ratio='', transform=None)
dataset = large_data_dist.large_data_dist('G:\DeepLearning\SiT_docker\dataset\large_data', split='trainval', is_pretrain=True, ratio='9999_1', transform=transform)
dataset_loader = DataLoader(dataset,
            batch_sampler=DistSampler(dataset, 512, 5),
            num_workers=0, pin_memory=True)
for i, (image, label) in enumerate(dataset_loader):
    print(len(label))
    break

    
