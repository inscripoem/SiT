import torch
import vision_transformer as vits
from main import FullPipline

student = vits.__dict__['vit_small'](img_size=[64], num_classes=2)



pretrain_dict = torch.load('./output/large_data_dist/checkpoint.pth')['student']
dict_to_load = {}
for k, v in pretrain_dict.items():
    if k.startswith('backbone'):
        dict_to_load[k.replace('backbone.', '')] = v

model_dict = student.state_dict()
model_dict.update(dict_to_load)
student.load_state_dict(model_dict)

for n, p in student.named_parameters():
    if not n.startswith('head'):
        p.requires_grad = False

student.cuda()
test_data = torch.randn(1, 3, 64, 64).cuda()
output = student(test_data, classify=True)
print(output)