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
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "./output/large_data_dist_classify/10000_10000/100/lr_5e-4/img_128/200_epoch" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--tensorboard_log_path "./output/large_data_dist_classify/10000_10000/100/lr_5e-4/img_128/tensorboard/200_epoch" `
--pretrain_model_path "./output/large_data_dist/img_128_patch_16/10000_10000/800_epoch/checkpoint.pth"
