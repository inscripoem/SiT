$DATASET = "large_data_dist"

$PETS_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet'
$LARGE_DATA_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\large_data'

if ($DATASET -eq "pets_dist") {
    $DATA_LOCATION = $PETS_DIST_LOCATION
} elseif ($DATASET -eq "large_data_dist") {
    $DATA_LOCATION = $LARGE_DATA_DIST_LOCATION
}

python main.py `
--batch_size 128 `
--patch_size 16 `
--epochs 800 `
--lmbda 5 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 2 `
--output_dir "./output/large_data_dist/img_128_patch_16/15000_5000" `
--image_size 128 `
--is_pretrain 1 `
--ratio '15000_5000' `
--tensorboard_log_path "./output/large_data_dist/img_128_patch_16/tensorboard/15000_5000"