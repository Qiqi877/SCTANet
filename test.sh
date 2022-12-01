python test.py --gpus 1 --model sparnet --name SCTANet --res_depth 24 --nfeat 180 --scale_factor 8 \
    --load_size 128 --dataset_name single --dataroot_test ./FACE_DATASET/Helen/Helen_test/LR/X8 \
    --pretrain_model_path ./check_points/SCTANet/best_net_G.pth \
    --save_as_dir ./results_helen/x8/ --test_psnr_ssim ./results-Helen-x8.txt --gt_dir ./FACE_DATASET/Helen/Helen_test/HR
