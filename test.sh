python test.py --gpus 1 --model sctanet --name test_checkpoint --res_depth 24 --nfeat 180 --scale_factor 8 \
    --load_size 128 --dataset_name single --dataroot_test /Celeba/img \
    --pretrain_model_path ./check_points/train_checkpoint/model_net_G.pth \
    --save_as_dir ./results_CelebA/ --test_psnr_ssim ./results-CelebA-x8.txt --gt_dir /Celeba/img

