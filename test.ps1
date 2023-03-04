$DATASET = "large_data_dist"

$PETS_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet'
$LARGE_DATA_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\large_data'

python main.py `
--batch_size 128 `
--epochs 200 `
--lmbda 5 `
--data_set $DATASET `
--data_location $LARGE_DATA_DIST_LOCATION `
--num_workers 2 `
--output_dir "./output/128_test" `
--image_size 128 `
--is_pretrain 1 `
--ratio '5000_5000'