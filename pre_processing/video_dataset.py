import logging

from pre_processing.video_reader import VideoReader
import random

from skimage.transform import resize
import cv2
import numpy as np
import h5py
import glob

import cvbase as cvb

HIGH_WIDTH = 256
HIGH_HEIGHT = 256

LOW_WIDTH = 64
LOW_HEIGHT = 64

FLOW_WIDTH = 64
FLOW_HEIGHT = 64

IMAGE_DEPTH = 3
FLOW_DEPTH = 3

TEMPORAL_STEPS = 10

GAUSSIAN_X_STD = 1.5
GAUSSIAN_Y_STD = 1.5


class VideoDataSet():
    def __init__(self, flow_net, directory, batch_size, frames_len, frame_max_try):
        self.flow_net = flow_net
        self.batch_size = batch_size
        self.frames_len = frames_len
        self.video_index = 0
        self.frame_max_try = frame_max_try
        self.frame_try = 0
        self.videos = get_files_in_dir(directory)
        self.current_video_reader = VideoReader(self.videos[0])
        self.current_frames = self.current_video_reader.read_batch(self.frames_len)
        self.current_down_scaled = down_scale_batch(self.current_frames, 2)


    def next_data(self):
        hr_batches, lr_batches = self.generate_random_batches(
            self.current_down_scaled,
            HIGH_WIDTH,
            HIGH_HEIGHT,
            self.batch_size,
            HIGH_WIDTH // LOW_WIDTH
        )
        flow_batches = calculate_batch_flows(lr_batches, self.flow_net)
        self.frame_try += 1
        self.update_current_frames_if_needed()
        return lr_batches, hr_batches, flow_batches


    def update_current_frames_if_needed(self):
        if self.frame_try == self.frame_max_try:
            self.frame_try = 0
            self.current_frames = self.current_video_reader.read_batch(self.frames_len)
            if len(self.current_frames) < self.frames_len:
                self.video_index = (self.video_index + 1) % len(self.videos)
                self.current_video_reader = VideoReader(self.videos[self.video_index])
                self.current_frames = self.current_video_reader.read_batch(self.frames_len)
            self.current_down_scaled = down_scale_batch(self.current_frames, 2)
        return


    def generate_random_batches(self, frames, new_width, new_height, number_of_batches, reduce_factor):
        hr_batches = []
        lr_batches = []
        for batch_idx in range(number_of_batches):
            dimensions = frames[0].shape
            start_w = random.randint(0, dimensions[0] - new_width)
            start_h = random.randint(0, dimensions[1] - new_height)
            lr_batch = []
            hr_batch = []
            for frame_idx in range(len(frames)):
                cur_frame = frames[frame_idx]
                cropped_image = cur_frame[start_w : start_w + new_width, start_h : start_h + new_height]
                blurred_image = cv2.GaussianBlur(cropped_image, (0, 0), GAUSSIAN_X_STD, GAUSSIAN_Y_STD, 0)
                lr_image = down_sample_image(blurred_image, reduce_factor)
                hr_batch.append(cropped_image)
                lr_batch.append(lr_image)
            hr_batches.append(hr_batch)
            lr_batches.append(lr_batch)
        return hr_batches, lr_batches



def get_files_in_dir(directory):
    files = [f for f in glob.glob(directory + "/*")]
    np.random.shuffle(files)
    return files


def down_scale_batch(frames, factor):
    down_scaled = []
    for frame_idx in range(len(frames)):
        dimensions = frames[frame_idx].shape
        (width, height) = dimensions[0] / factor, dimensions[1] / factor
        down_scaled.append(resize(frames[frame_idx], (width, height)))
    return down_scaled


def calculate_batch_flows(lr_batches, flow_net):
    flow_batches = []
    for lr_batch in lr_batches:
        flow_batches.append(calculate_flow(lr_batch, flow_net))
    return flow_batches


def calculate_flow(low_batch, flow_net):
    flow_results = []
    for idx in range(len(low_batch)):
        if idx == 0:
            low_1 = np.zeros((LOW_WIDTH, LOW_HEIGHT, IMAGE_DEPTH))
        else:
            low_1 = low_batch[idx - 1]
        low_2 = low_batch[idx]
        flow_results.append(flow_net.inference_imgs(low_1, low_2))
    return flow_results


def down_sample_image(image, factor):
    new_image = np.zeros((int(image.shape[0] / factor), int(image.shape[1] / factor), 3))
    row = factor - 1
    while row < image.shape[0]:
        col = factor - 1
        while col < image.shape[1]:
            new_image[int(row / factor), int(col / factor), :] = image[row, col, :]
            col += factor
        row += factor
    return new_image
