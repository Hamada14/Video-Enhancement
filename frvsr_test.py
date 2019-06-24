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


def trunc(tensor):
    # tensor = tensor.clone()
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    return tensor

def clip_output_img(hr_out):
    hr_out = trunc(hr_out.clone())
    hr_out = hr_out.cpu()
    out_img = hr_out.data[0].numpy()
    out_img *= 255.0
    return (np.uint8(out_img)).transpose((1, 2, 0))

def clip_lr(hr_out):
    hr_out = trunc(hr_out.clone())
    hr_out = hr_out.cpu()
    out_img = hr_out.data[0].numpy()
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
    return lr_frame


if __name__ == "__main__":
    with torch.no_grad():
        parser = argparse.ArgumentParser(description='Test Single Video')
        parser.add_argument('--video', type=str, help='test low resolution video name')
        parser.add_argument('--model', type=str, help='generator model epoch name')
        parser.add_argument('--output', type=str, help='output video path')
        parser.add_argument('--lr', type=str, help='low resolution video path')
        parser.add_argument('--hr', type=str, help='high resolution video path')

        opt = parser.parse_args()

        UPSCALE_FACTOR = 4
        VIDEO_NAME = opt.video
        MODEL_NAME = opt.model
        OUTPUT_VIDEO = opt.output
        LR_VIDEO = opt.lr
        HR_VIDEO = opt.hr
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        videoCapture = cv2.VideoCapture(VIDEO_NAME)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        hr_width = int(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) / (UPSCALE_FACTOR * 64) ) * (64 * UPSCALE_FACTOR)
        hr_height = int(int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) / (UPSCALE_FACTOR * 64) ) * (64 * UPSCALE_FACTOR)
        lr_width = int(hr_width / UPSCALE_FACTOR)
        lr_height = int(hr_height / UPSCALE_FACTOR)

        sr_video_size = (hr_width, hr_height)

        lr_video_size = (lr_height, lr_height)

        sr_video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, sr_video_size)
        lr_video_writer = cv2.VideoWriter(LR_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, lr_video_size)
        hr_video_writer = cv2.VideoWriter(HR_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, sr_video_size)

        success,hr_frame = videoCapture.read()
        test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')

        model = FRVSR_models.FRVSR(0, 0, 0)
        model.to(device)
        model.load_state_dict(torch.load(MODEL_NAME, device))
        model.eval()
        
        model.set_param(batch_size=1, width=lr_width, height=lr_height)
        model.init_hidden(device)

        for index in test_bar:
            if index > 1200:
                break
            if success:
                hr_frame = hr_frame[0:hr_height, 0:hr_width, :]
                lr_frame = hr_to_lr(hr_frame)
                lr_image = Variable(ToTensor()(lr_frame)).unsqueeze(0).float()
                lr_image.to(device)
                if torch.cuda.is_available():
                    lr_image = lr_image.cuda()
                est_hr, lr_out = model(lr_image)
                sr_video_writer.write(clip_output_img(est_hr))
                lr_video_writer.write(clip_lr(ToTensor()(lr_frame)))
                hr_video_writer.write(hr_frame)
                success, hr_frame = videoCapture.read()
        sr_video_writer.release()
        lr_video_writer.release()
        hr_video_writer.release()
