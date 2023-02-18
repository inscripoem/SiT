import os
import sys

from datasets.Pets import pets
from datasets.Pets_dist import pets_dist
from datasets.large_data_dist import large_data_dist

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):


    if args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        
        nb_classes = 37

    elif args.data_set == 'Pets_dist':
        split = 'trainval' if is_train else 'test'
        dataset = pets_dist(os.path.join(args.data_location), split=split, is_pretrain=args.is_pretrain, ratio=args.ratio, transform=trnsfrm)

        nb_classes = 2
    
    elif args.data_set == 'large_data_dist':
        split = 'trainval' if is_train else 'test'
        dataset = large_data_dist(os.path.join(args.data_location), split=split, ratio=args.ratio, transform=trnsfrm)

        nb_classes = 2

    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
       

        
        
    return dataset, nb_classes


