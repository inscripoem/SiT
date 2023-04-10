$DATASET = "large_data_dist"

$PETS_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet'
$LARGE_DATA_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\large_data'

if ($DATASET -eq "pets_dist") {
    $DATA_LOCATION = $PETS_DIST_LOCATION
} elseif ($DATASET -eq "large_data_dist") {
    $DATA_LOCATION = $LARGE_DATA_DIST_LOCATION
}

python main.py `
--batch_size 256 `
--epochs 200 `
--drop_path_rate 0.2 `
--warmup_epochs 10 `
--lr 0.0001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "./output/large_data_dist_classify/from_scratch/100/lr_7e-4_0.2/img_128/200_epoch" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--tensorboard_log_path "./output/large_data_dist_classify/from_scratch/100/lr_7e-4_0.2/img_128/tensorboard/200_epoch"
