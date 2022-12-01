import os
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
from psnr_ssim import psnr_ssim_dir

def val(model,opt):
    gt_dir = opt.dataroot_test

    tmp_phase = opt.phase
    tmp_num_threads = opt.num_threads
    tmp_batch_size = opt.batch_size
    tmp_serial_batches = opt.serial_batches

    opt.phase = 'test'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    if len(opt.save_as_dir_test):
        save_dir = opt.save_as_dir_test
    os.makedirs(save_dir, exist_ok=True)

    print('creating result directory', save_dir)

    network = model.netG
    network.eval()

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        inp = data['LR']
        with torch.no_grad():
            output_SR = network(inp)
        img_path = data['HR_paths']     # get image paths
        output_sr_img = utils.tensor_to_img(output_SR, normal=True)

        save_path = os.path.join(save_dir, img_path[0].split('/')[-1]) 
        save_img = Image.fromarray(output_sr_img)
        save_img.save(save_path)

    result = psnr_ssim_dir(save_dir, gt_dir)

    opt.phase = tmp_phase
    opt.num_threads = tmp_num_threads
    opt.batch_size = tmp_batch_size
    opt.serial_batches = tmp_serial_batches

    return result
       
