import warnings
warnings.filterwarnings("ignore")


import argparse
import os
import sys
import datetime
import time
import math
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, datasets_utils
from sit import SiT
from classify import Classify

import utils
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead
from torchsummaryX import summary

from thop import profile

def get_args_parser(parser: argparse.ArgumentParser = None):
    # Reconstruction Parameters
    parser.add_argument('--drop_perc', type=float, default=0.5, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches; Set to patch size to align corruption with patches')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    
    parser.add_argument('--lmbda', type=int, default=1, help='Scaling factor for the reconstruction loss\n')
    
    # SimCLR Parameters
    parser.add_argument('--out_dim', default=256, type=int, help="Dimensionality of output features")
    parser.add_argument('--simclr_temp', default=0.2, type=float, help="tempreture for SimCLR.")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="EMA parameter for teacher update.\n")
    

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate\n")
    

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=800, type=int, help='Number of epochs of training.')

    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.\n")
    

    # Dataset
    parser.add_argument('--data_set', default='Pets', type=str, 
                        choices=['STL10', 'MNIST', 'CIFAR10', 'CIFAR100', 'Flowers', 'Aircraft', 
                                 'Cars', 'ImageNet5p', 'ImageNet', 'TinyImageNet', 'Pets', 'Pets_dist', 'CUB', 'large_data_dist',
                                 'PASCALVOC', 'MSCOCO', 'VisualGenome500'], 
                        help='Name of the dataset.')
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')

    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--image_size', default=64, type=int, help='Image size.')

    parser.add_argument('--is_pretrain', default=True, type=utils.bool_flag, help="Define if it's pre-train.")
    parser.add_argument('--ratio', default="", type=str, help="The distribution ratio(neg_pos)/label usage of the dataset(percentage).")
    parser.add_argument('--pretrain_model_path', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--pretrain_adjust_mode', type=str, default='linear', help='How to train the downstream task')
    parser.add_argument('--tensorboard_log_path', type=str, default='', help='Path to tensorboard log')
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size of ViT model")
    return parser

# replace from other images


def main():
    parser = argparse.ArgumentParser('SiT')
    parser = get_args_parser(parser)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set the seed
    utils.fix_random_seeds(args.seed)
    
    # Check if gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
        cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    # Load the dataset
    transform = datasets_utils.DataAugmentationSiT(args)
    if args.is_pretrain:
        train_dataset = load_dataset.build_dataset(args, split='train', trnsfrm=transform)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
            shuffle=True, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=False, 
            collate_fn=datasets_utils.collate_batch(args.drop_replace, args.drop_align))
    else:
        transform_train = datasets_utils.transform_train(args).transform
        train_dataset = load_dataset.build_dataset(args, split='train', trnsfrm=transform_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
            shuffle=True, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        transform_valtest = datasets_utils.transform_test(args).transform
        val_dataset = load_dataset.build_dataset(args, split='val', trnsfrm=transform_valtest)
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
            shuffle=True, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        test_dataset = load_dataset.build_dataset(args, split='test', trnsfrm=transform_valtest)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
            shuffle=True, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Create models
    example_input = torch.rand(1, 3, args.image_size, args.image_size)
    if args.is_pretrain:
        student = vits.__dict__[args.model](drop_path_rate=args.drop_path_rate, img_size=[args.image_size], patch_size=args.patch_size)
        teacher = vits.__dict__[args.model](img_size=[args.image_size], patch_size=args.patch_size)
        embed_dim = student.embed_dim

        student = FullPipline(student, CLSHead(embed_dim, args.out_dim), RECHead(embed_dim, patch_size=args.patch_size))
        teacher = FullPipline(teacher, CLSHead(embed_dim, args.out_dim), RECHead(embed_dim, patch_size=args.patch_size))
        
        student_copy = copy.deepcopy(student)
        flops, params = profile(student_copy, (example_input,))
        #df = summary(student_copy, example_input)

        student, teacher = student.to(device), teacher.to(device)
        # teacher and student start with the same parameters
        teacher.load_state_dict(student.state_dict())

        # fix teacher parameters
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {args.model} network.")
        print(f"Student FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")

    else:
        student = vits.__dict__[args.model](drop_path_rate=args.drop_path_rate, img_size=[args.image_size], num_classes=2, patch_size=args.patch_size)
        
        student_copy = copy.deepcopy(student)
        flops, params = profile(student_copy, (example_input,))
        #df = summary(student_copy, example_input)

        student = student.to(device)

        print(f"Model is built: it is {args.model} network.")
        print(f"Model FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
    with open(Path(args.output_dir) / "model.txt", "w") as f:
        f.write(json.dumps({"flops": f'{flops/1e9:.2f}G', "params": f'{params/1e6:.2f}M',}))
        #f.write(json.dumps(df, indent=4))
    
    # Define optimizer
    optimizer = torch.optim.AdamW(utils.get_params_groups(student))

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Define schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr, 
        args.min_lr, args.epochs, len(train_data_loader), warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler( args.weight_decay,
        args.weight_decay_end, args.epochs, len(train_data_loader))
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(train_data_loader))

    # Resume training 
    to_restore = {"epoch": 0, "lr_schedule": lr_schedule, "wd_schedule": wd_schedule, "momentum_schedule": momentum_schedule}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher if args.is_pretrain else None,
        optimizer=optimizer, fp16_scaler=fp16_scaler)
    
    start_epoch = to_restore["epoch"]
    lr_schedule = to_restore["lr_schedule"]
    wd_schedule = to_restore["wd_schedule"]
    momentum_schedule = to_restore["momentum_schedule"]

    start_time = time.time()
    print(f"Start training {'pretrain' if args.is_pretrain else 'classify'} model at the ratio of {args.ratio}{'%' if not args.is_pretrain else ''}.")

    if args.is_pretrain:
        sit = SiT(args, device, student, teacher, optimizer, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule)
        sit.train(start_epoch, train_data_loader)
    else:
        classify = Classify(args, device, student, optimizer, fp16_scaler, lr_schedule, wd_schedule)
        classify.main(start_epoch, train_data_loader, val_dataset_loader, test_dataset_loader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

        
        
class FullPipline(nn.Module):
    def __init__(self, backbone, head, head_recons):
        super(FullPipline, self).__init__()

        
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons

    def forward(self, x, recons=True):
        _out = self.backbone(x)
        
        if recons==True:
            return self.head(_out[:, 0]), self.head_recons(_out[:, 1:])
        else:
            return self.head(_out[:, 0]), None

if __name__ == '__main__':
    main()