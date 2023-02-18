YELLOW1="\033[33m"
YELLOW2="\033[0m"

DATASET="Pets_dist"

RATIO1="4799_1"
RATIO2="4600_200"
RATIO3="2370_2370"

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
    TRAIN_RATIO=RATIO$i
    echo -e "${YELLOW1}Start training ratio ${!TRAIN_RATIO}......${YELLOW2}"
    python -m torch.distributed.launch --nproc_per_node=1 --use_env main_test.py --batch_size 8 --epochs 200 --data_set $DATASET --data_location "/app/dataset/Pets_dataset/oxford-iiit-pet" --num_workers 2 --is_pretrain True --ratio ${!TRAIN_RATIO} --output_dir "/app/output/${DATASET}_ratio_${!TRAIN_RATIO}" | tee logs/${MONTH}/${DAY}/train_${DATASET}_${!TRAIN_RATIO}_at_${HOUR}_${MINUTE}_${SECOND}.log
    echo -e "${YELLOW1}Training complete.${YELLOW2}"
done