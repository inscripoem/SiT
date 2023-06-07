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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomResizedCrop(224),
        ])
dataset = Pets_dist.pets_dist('G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet', split='train', is_pretrain=False, ratio='5', transform=transform)
#dataset = large_data_dist.large_data_dist('G:\DeepLearning\SiT_docker\dataset\large_data', split='train', is_pretrain=False, ratio='5', transform=transform)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)  

print(len(dataloader))
