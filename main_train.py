import warnings
warnings.filterwarnings("ignore")


import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import load_dataset, datasets_utils

import utils
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead
import torchvision


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
                                 'Cars', 'ImageNet5p', 'ImageNet', 'TinyImageNet', 'Pets', 'Pets_dist', 'CUB',
                                 'PASCALVOC', 'MSCOCO', 'VisualGenome500'], 
                        help='Name of the dataset.')
    parser.add_argument('--data_location', default='/path/to/dataset', type=str, help='Dataset location.')

    parser.add_argument('--output_dir', default="checkpoints/vit_small/trial", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument('--is_pretrain', default=True, type=utils.bool_flag, help="Define if it's pre-train.")
    parser.add_argument('--ratio', default="", type=str, help="The distribution ratio(neg_pos)/label usage of the dataset(percentage).")
    return parser

# replace from other images
class collate_batch(object): 
    def __init__(self, drop_replace=0., drop_align=1):
        self.drop_replace = drop_replace
        self.drop_align = drop_align
        
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        
        if self.drop_replace > 0:
            batch[0][1][0], batch[0][2][0] = datasets_utils.GMML_replace_list(batch[0][0][0], batch[0][1][0], batch[0][2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[0][1][1], batch[0][2][1] = datasets_utils.GMML_replace_list(batch[0][0][1], batch[0][1][1], batch[0][2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
        
        return batch
    

def train_SiT(args):
    utils.fix_random_seeds(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT')
    parser = get_args_parser(parser)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
