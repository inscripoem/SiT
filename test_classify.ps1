$DATASET = "large_data_dist"

$PETS_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\Pets_dataset\oxford-iiit-pet'
$LARGE_DATA_DIST_LOCATION = 'G:\DeepLearning\SiT_docker\dataset\large_data'

python main.py `
--batch_size 512 `
--epochs 200 `
--lmbda 5 `
--data_set $DATASET `
--data_location $LARGE_DATA_DIST_LOCATION `
--num_workers 2 `
--output_dir "./output/large_data_dist_classify" `
--image_size 64 `
--is_pretrain 0 `
--ratio '10' `
--pretrain_model_path './output/large_data_dist/checkpoint.pth'