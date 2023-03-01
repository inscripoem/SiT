
from datasets import Pets_dist, large_data_dist

dataset = Pets_dist.pets_dist('G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet', split='test', is_pretrain=False, ratio='', transform=None)
dataset = large_data_dist.large_data_dist('G:\DeepLearning\SiT_docker\dataset\large_data', split='trainval', is_pretrain=True, ratio='9999_1', transform=None)
a, b = dataset.__getitem__(0)
print(b)