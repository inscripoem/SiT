DAY=`date +%d`
MONTH=`date +%m`
HOUR=`date +%H`
MINUTE=`date +%M`
SECOND=`date +%S`

DATASET="data_test"

python main.py --batch_size 128 --epochs 30 --lmbda 5 --data_set "ImageNet" --data_location "/app/dataset/$DATASET" --num_workers 2 --output_dir "/app/output/$DATASET"