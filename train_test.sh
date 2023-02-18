DAY=`date +%d`
MONTH=`date +%m`
HOUR=`date +%H`
MINUTE=`date +%M`
SECOND=`date +%S`

YELLOW1="\033[33m"
YELLOW2="\033[0m"

if [ ! -d logs/${MONTH}/${DAY} ]; then
  mkdir -p logs/${MONTH}/${DAY};
fi

DATASET="data_test"

python -m torch.distributed.launch --nproc_per_node=1 --use_env main_test.py --batch_size 16 --epochs 30 --data_set "ImageNet" --data_location "/app/dataset/$DATASET" --num_workers 2 --output_dir "/app/output/$DATASET" | tee logs/${MONTH}/${DAY}/train_${DATASET}_at_${HOUR}_${MINUTE}_${SECOND}.log