import logging

from frvsr.pre_processing.video_reader import VideoReader
from frvsr.pre_processing.cell_input import CellInput
import random

from skimage.transform import resize
import cv2
import numpy as np
import h5py

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
BATCH_SIZE = 10

GAUSSIAN_X_STD = 1.5
GAUSSIAN_Y_STD = 1.5

logger = logging.getLogger()

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

def generate_random_batches(frames, new_width, new_height, number_of_batches, reduce_factor):
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

def down_scale_batch(frames, factor):
    down_scaled = []
    for frame_idx in range(len(frames)):
        dimensions = frames[frame_idx].shape
        (width, height) = dimensions[0] / factor, dimensions[1] / factor
        down_scaled.append(resize(frames[frame_idx], (width, height)))

    return down_scaled

def store_to_file(dataset, data_tuples):
    dataset_len = dataset.len()
    data_tuples_len = len(data_tuples)
    dataset.resize(dataset_len + data_tuples_len, axis=0)
    dataset[-data_tuples_len:] = data_tuples

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

def calculate_batch_flows(lr_batches, flow_net):
    flow_batches = []
    for lr_batch in lr_batches:
        flow_batches.append(calculate_flow(lr_batch, flow_net))
    return flow_batches

def tranform_video(source_video_path, flow_net):
    logger.info('Transforming video {' + source_video_path+ '}')
    logger.info('Successfully created the destination file')
    logger.info('Reading the video file')
    reader = VideoReader(source_video_path)
    frames = reader.read_batch(10)
    i = 0
    while (len(frames) >= 10):
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Read the input batch successfully')
        down_scaled = down_scale_batch(frames, 2)
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Downscaled the frames successfully')
        hr_batches, lr_batches = generate_random_batches(down_scaled, HIGH_WIDTH, HIGH_HEIGHT, BATCH_SIZE, HIGH_WIDTH // LOW_WIDTH)
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Generated random batches from downscaled frames')
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Initializing flow calculation')
        flow_path_batches = calculate_batch_flows(lr_batches, flow_net)
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Finished calculating the flow for the current batch')
        # Call to the model
        axarr[0,0].imshow(lr_batches[batch_index][img1_index])
        axarr[0,1].imshow(lr_batches[batch_index][img2_index])
        axarr[1,0].imshow(hr_batches[batch_index][img1_index])
        axarr[1,1].imshow(hr_batches[batch_index][img2_index])

        plt.show()
        # train the SR model
        logger.info('[batch: ' + str(i + 1) + '] ' + 'Appended the batch to the dataset file')
        frames = reader.read_batch(10)
        i += 1

#     high: nx10x10x256x256x3
#     low: nx10x10x64x64x3
#     flow: nx10x10x64x64x2
#     number x random x seq of ten x image_dim
