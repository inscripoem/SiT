import os
import sys

from datasets.Pets import pets
from datasets.Pets_dist import pets_dist
from datasets.large_data_dist import large_data_dist

def build_dataset(args, split, trnsfrm=None, training_mode='finetune'):


    if args.data_set == 'Pets':
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        

    elif args.data_set == 'Pets_dist':
        dataset = pets_dist(os.path.join(args.data_location), split=split, is_pretrain=args.is_pretrain, ratio=args.ratio, transform=trnsfrm)

    
    elif args.data_set == 'large_data_dist':
        dataset = large_data_dist(os.path.join(args.data_location), split=split, is_pretrain=args.is_pretrain, ratio=args.ratio, transform=trnsfrm)


    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
       

        
        
    return dataset


