import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import frvsr.FRVSR_models as FRVSR_models
    
import logging
from pre_processing.video_reader import VideoReader
import random
from skimage.transform import resize
import cv2
import numpy as np
import h5py
import glob
from torchvision import datasets, transforms
import cvbase as cvb
import math
import os

def trunc(tensor):
    # tensor = tensor.clone()
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    return tensor

PIXEL_MAX = 255.0
 
def image_psnr(image_1, image_2):
    mse = np.mean( (image_1 - image_2) ** 2 )
    if(mse <= 0.001):
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def clip_output_img(hr_out):
    hr_out = trunc(hr_out.clone())
    hr_out = hr_out.cpu()
    out_img = hr_out.data[0].numpy()
    out_img *= 255.0
    return (np.uint8(out_img)).transpose((1, 2, 0))

def clip_lr(out_img):
    out_img *= 255.0
    return (np.uint8(out_img))

def clip_hr(hr_out):
    out_img = hr_out
    out_img *= 255.0
    return (np.uint8(out_img))


def downsample_frame(image, factor):
    new_image = np.zeros((int(image.shape[0] / factor), int(image.shape[1] / factor), 3))
    row = factor - 1
    while row < image.shape[0]:
        col = factor - 1
        while col < image.shape[1]:
            new_image[int(row / factor), int(col / factor), :] = image[row, col, :]
            col += factor
        row += factor
    return new_image


def hr_to_lr(hr_frame):
    blurred_frame = cv2.GaussianBlur(hr_frame, (0, 0), 1.5, 1.5, 0)
    lr_frame = downsample_frame(blurred_frame, 4)
    return np.clip(lr_frame/255.0, a_min = 0, a_max = 1)


def get_files_in_dir(path):

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    print(files)
    return files


def read_img(file_path, is_low):
    img = cv2.imread(file_path)
    if is_low:
        ch = int(img.shape[0] / 64)  * 64
        cw = int(img.shape[1] / 64) * 64
        return img[:ch, :cw, :]
    ch = int(img.shape[0] / (64 * 4)) * 64 * 4
    cw = int(img.shape[1] / (64 * 4)) * 64 * 4
    return img[:ch, :cw, :]
    
if __name__ == "__main__":
    with torch.no_grad():
        parser = argparse.ArgumentParser(description='Test Single Video')
        parser.add_argument('--dir', type=str, help='test low resolution video name')
        parser.add_argument('--model', type=str, help='generator model epoch name')

        opt = parser.parse_args()

        UPSCALE_FACTOR = 4
        DIR = opt.dir
        MODEL_NAME = opt.model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        lr_files = get_files_in_dir(DIR + "LR/")
        lr_temp = read_img(lr_files[0], True)

        model = FRVSR_models.FRVSR(0, 0, 0)
        model.to(device)
        model.load_state_dict(torch.load(MODEL_NAME, device))
        model.eval()
        
        model.set_param(batch_size=1, width=lr_temp.shape[1], height=lr_temp.shape[0])
        model.init_hidden(device)
        i = 0
        psnr_sum = 0

        test_bar = tqdm(range(len(lr_files)))
        for index in test_bar:
            lr_frame = read_img(lr_files[index], True)
            hr_path = DIR + "HR/" + lr_files[index][len(DIR) + 3:]
            print(hr_path)
            print(lr_files[index])
            hr_frame = read_img(hr_path, False)
            try:
                model.set_param(width = lr_frame.shape[1], height=lr_temp.shape[0])
                model.init_hidden(device)
            except:
                continue
            lr_image = Variable(ToTensor()(lr_frame)).unsqueeze(0).float()
            lr_image.to(device)
            lr_image = lr_image.cuda()
            try:
                est_hr, lr_out = model(lr_image)
            except:
                continue
            psnr_val = image_psnr(hr_frame, clip_output_img(est_hr))
            psnr_sum = psnr_sum + psnr_val
            print(psnr_val)
            i = i + 1
        print("Average PSNR is ", psnr_sum / i, " for images ", i, "/", len(lr_files))
