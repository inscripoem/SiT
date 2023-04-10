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
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\10000_10000\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\10000_10000\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\15000_5000\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\15000_5000\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\17500_2500\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\17500_2500\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\18750_1250\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\18750_1250\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\19750_250\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19750_250\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\19950_50\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19950_50\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\19990_10\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19990_10\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"

python main.py `
--batch_size 256 `
--epochs 50 `
--warmup_epochs 10 `
--lr 0.00001 `
--data_set $DATASET `
--data_location $DATA_LOCATION `
--num_workers 4 `
--output_dir "G:\DeepLearning\SiT_docker\output\large_data_dist_classify\finetuning\19998_2\lr_1e-5_img_128_epoch_50\100" `
--image_size 128 `
--is_pretrain 0 `
--ratio '100' `
--pretrain_model_path "G:\DeepLearning\SiT_docker\output\large_data_dist\img_128_patch_16\19998_2\checkpoint.pth" `
--pretrain_adjust_mode "finetuning"