import torch
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead
from main import FullPipline
from thop import profile

patch_size = 16

example_input = torch.rand(1, 3, 128, 128)
student = vits.__dict__['vit_small'](img_size=[128], num_classes=2, patch_size=patch_size)
embed_dim = student.embed_dim

student = FullPipline(student, CLSHead(embed_dim, 256), RECHead(embed_dim, patch_size=patch_size))

flops, params = profile(student, (example_input,))
print(f'Image size: 128x128, Patch size: {patch_size}x{patch_size}')
print(f"Student FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")

'''
pretrain_dict = torch.load('./output/large_data_dist/img_128_patch_16/10000_10000/800_epoch/checkpoint.pth')['student']
dict_to_load = {}
dict_not_load = {}
for k, v in pretrain_dict.items():
    if k.startswith('backbone'):
        if v.shape == student.state_dict()[k.replace('backbone.', '')].shape:
            dict_to_load[k.replace('backbone.', '')] = v
        else:
            dict_not_load[k.replace('backbone.', '')] = v

print(dict_to_load.keys())
print(dict_not_load.keys())

model_dict = student.state_dict()
model_dict.update(dict_to_load)
student.load_state_dict(model_dict)

for n, p in student.named_parameters():
    print(n, p.shape)
    if not n.startswith('head'):
        p.requires_grad = False

student.cuda()
test_data = torch.randn(1, 3, 64, 64).cuda()
output = student(test_data, classify=True)
print(output)
'''