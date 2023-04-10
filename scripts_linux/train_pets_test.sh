YELLOW1="\033[33m"
YELLOW2="\033[0m"

DATASET="Pets_dist"

RATIO="2370_2370"

DAY=`date +%d`
MONTH=`date +%m`
HOUR=`date +%H`
MINUTE=`date +%M`
SECOND=`date +%S`

if [ ! -d logs/${MONTH}/${DAY} ]; then
  mkdir -p logs/${MONTH}/${DAY};
fi
echo -e "${YELLOW1}Start training ratio ${RATIO}......${YELLOW2}"
python main_train.py --batch_size 512 --epochs 800 --lmbda 5 --data_set $DATASET --data_location "/app/dataset/Pets_dataset/oxford-iiit-pet" --num_workers 2 --is_pretrain True --ratio ${RATIO} --output_dir "/app/output/${DATASET}_ratio_${RATIO}" | tee logs/${MONTH}/${DAY}/train_${DATASET}_${RATIO}_at_${HOUR}_${MINUTE}_${SECOND}.log
echo -e "${YELLOW1}Training complete.${YELLOW2}"
