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
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\10000_10000\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\10000_10000\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\15000_5000\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\15000_5000\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\17500_2500\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\17500_2500\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\18750_1250\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\18750_1250\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\19750_250\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19750_250\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\19950_50\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19950_50\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\19990_10\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19990_10\checkpoint.pth"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.0005 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\linear_evaluation\19998_2\lr_5e-4_img_128_epoch_50\50" `
--image_size 128 `
--is_pretrain 0 `
--ratio '50' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19998_2\checkpoint.pth"