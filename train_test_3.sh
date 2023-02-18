DATASET1="data_neg_9995_pos_5"
DATASET2="data_neg_9375_pos_625"
DATASET3="data_neg_7500_pos_2500"

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
    python -m torch.distributed.launch --nproc_per_node=1 --use_env main_test.py --batch_size 16 --epochs 400 --data_set "ImageNet" --data_location "/app/dataset/${!TRAIN_DATASET}" --num_workers 4 --output_dir "/app/output/${!TRAIN_DATASET}" | tee logs/${MONTH}/${DAY}/train_${!TRAIN_DATASET}_at_${HOUR}_${MINUTE}_${SECOND}.log
    echo -e "${YELLOW1}Training complete.${YELLOW2}"
done