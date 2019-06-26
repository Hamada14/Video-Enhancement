import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import logging

from srgans.recurrent_srgan_model_test import RecurrentSRGAN 
from flow_model_wrapper import FlowModelWrapper

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
import tensorflow as tf



def clip_output_img(hr_out):
    #hr_out = trunc(hr_out.clone())
    out_img = np.copy(hr_out)
    out_img = (out_img - np.min(out_img)) / np.ptp(out_img)
    #hr_out = hr_out.cpu()
    out_img = out_img[0][0]
    out_img *= 255.0
    return (np.uint8(out_img))

def clip_lr(lr_out):
    #hr_out = trunc(hr_out.clone())
    #hr_out = hr_out.cpu()
    out_img = np.copy(lr_out)
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
    #blurred_frame = cv2.GaussianBlur(hr_frame, (0, 0), 1.5, 1.5, 0)
    lr_frame = downsample_frame(hr_frame, 4)
    return lr_frame


if __name__ == "__main__":
    print('hey')
    with torch.no_grad():
        print("started")
#        parser = argparse.ArgumentParser(description='Test Single Video')
#        parser.add_argument('video', type=str, help='test low resolution video name')
       # parser.add_argument('--model', type=str, help='generator model epoch name')
#        parser.add_argument('output', type=str, help='output video path')
#        parser.add_argument('lr', type=str, help='low resolution video path')
#        parser.add_argument('hr', type=str, help='high resolution video path')

#        opt = parser.parse_args()

        UPSCALE_FACTOR = 4
        VIDEO_NAME =  '/home/ubuntu/Video-Enhancement/data_set/Sample1280.mp4'                                                                                          
        #MODEL_NAME = opt.model
        OUTPUT_VIDEO = '/home/ubuntu/Video-Enhancement/output_gan_without_blur.mp4'
        LR_VIDEO = '/home/ubuntu/Video-Enhancement/low1.mp4'
        HR_VIDEO = '/home/ubuntu/Video-Enhancement/high1.mp4'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        videoCapture = cv2.VideoCapture(VIDEO_NAME)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        hr_width = int(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) / (UPSCALE_FACTOR * 64) ) * (64 * UPSCALE_FACTOR)
        hr_height = int(int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) / (UPSCALE_FACTOR * 64) ) * (64 * UPSCALE_FACTOR)
        lr_width = int(hr_width / UPSCALE_FACTOR)
        lr_height = int(hr_height / UPSCALE_FACTOR)

        sr_video_size = (hr_width, hr_height)

        lr_video_size = (lr_width, lr_height)

        sr_video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, sr_video_size)
        lr_video_writer = cv2.VideoWriter(LR_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, lr_video_size)
        hr_video_writer = cv2.VideoWriter(HR_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, sr_video_size)

        success,hr_frame = videoCapture.read()
        test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]') 
        model = RecurrentSRGAN(high_width = hr_width, high_height = hr_height, low_width = lr_width, low_height = lr_height, batch_size = 1, time_steps = 1)
        flow_model = FlowModelWrapper.getInstance()
        initial_hr_estimate = np.zeros([1, hr_height, hr_width, 3])
        initial_lr_estimate = np.zeros([lr_height, lr_width, 3])

        lpips_metric = 0
        psnr_metric = 0

        for index in test_bar:
            if index > 1200:
                break
            if success:
                hr_frame = hr_frame[0:hr_height, 0:hr_width, :]
                lr_frame = hr_to_lr(hr_frame)
                lr_image = np.expand_dims(np.expand_dims(lr_frame, axis=0), axis=0)
                #lr_image 
                if (index == 0):
                   prev_est = initial_hr_estimate
                   flow_estimate = flow_model.inference_imgs(initial_lr_estimate, lr_frame)
                   
                else:
                    flow_estimate = flow_model.inference_imgs(prev_input[0][0], lr_frame)
                flow_estimate = np.expand_dims(np.expand_dims(flow_estimate , axis=0), axis=0)
                est_hr, prev_input = model.estimate_frames(prev_est, lr_image, flow_estimate)
                sr_video_writer.write(clip_output_img(est_hr))
                lr_video_writer.write(clip_lr((lr_frame)))
                hr_video_writer.write(hr_frame)

                estimated_hr_norm = (est_hr - np.min(est_hr)) / np.ptp(est_hr)
                lpips_metric += model.evaluate_with_lpips_metric(estimated_hr_norm, hr_frame)
                psnr_metric += model.evaluate_with_psnr_metric(estimated_hr_norm, hr_frame)

                success, hr_frame = videoCapture.read()
                prev_est = est_hr[0]
        sr_video_writer.release()
        lr_video_writer.release()
        hr_video_writer.release()

        logging.info("LPIPS %.8f " % (lpips_metric/frame_numbers))
        logging.info("PSNR %.8f " % (psnr_metric / frame_numbers))

