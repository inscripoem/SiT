DATASET1="data_neg_9999_pos_1"
DATASET2="data_neg_9875_pos_125"
DATASET3="data_neg_5000_pos_5000"

YELLOW1="\033[33m"
YELLOW2="\033[0m"

for i in {1..3}
do
    DAY=`date +%d`
    MONTH=`date +%m`
    HOUR=`date +%H`
    MINUTE=`date +%M`
    SECOND=`date +%S`
    
    if [ ! -d logs/${MONTH}/${DAY} ]; then
      mkdir -p logs/${MONTH}/${DAY};
    fi

    echo -e "${YELLOW1}Start training dataset $i......${YELLOW2}"
    TRAIN_DATASET=DATASET$i
    python -m torch.distributed.launch --nproc_per_node=1 --use_env main_test.py --batch_size 256 --epochs 400 --lmbda 5 --data_set "ImageNet" --data_location "/app/dataset/${!TRAIN_DATASET}" --num_workers 4 --output_dir "/app/output/${!TRAIN_DATASET}" | tee logs/${MONTH}/${DAY}/train_${!TRAIN_DATASET}_at_${HOUR}_${MINUTE}_${SECOND}.log
    echo -e "${YELLOW1}Training complete.${YELLOW2}"
done