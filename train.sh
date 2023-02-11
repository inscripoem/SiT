TIME=`date | cut --delimiter=" " --fields=4`
DAY=`date | cut --delimiter=" " --fields=3`
HOUR=`echo $TIME | cut --delimiter=":" --fields=1`
MINUTE=`echo $TIME | cut --delimiter=":" --fields=2`
SECOND=`echo $TIME | cut --delimiter=":" --fields=3`

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 8 --epochs 801 --data_set "ImageNet" --data_location '/app/dataset/data_3_neg_5000_pos_2500' --num_workers 2 --output_dir '/app/output/data_3_neg_5000_pos_2500' | tee logs/train_${DAY}_${HOUR}_${MINUTE}_${SECOND}.log