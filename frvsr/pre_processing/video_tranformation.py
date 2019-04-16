from video_reader import VideoReader
from cell_input import CellInput
import random

from skimage.transform import resize
import cv2
import numpy as np
import h5py
from matplotlib import pyplot as plt



def down_sample_image(image, factor):
    new_image = np.zeros((int(image.shape[0] / factor), int(image.shape[1] / factor), 3))
    row = factor - 1
    while (row < image.shape[0]):
        col = factor - 1
        while (col < image.shape[1]):
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
        lr_batch.append(np.zeros((int(new_height / reduce_factor), int(new_width / reduce_factor), 3)))
        for frame_idx in range(len(frames)):
            cur_frame = frames[frame_idx]
            cropped_image = cur_frame[start_w : start_w + new_width, start_h : start_h + new_height]
            blurred_image = cv2.GaussianBlur(cropped_image, (0, 0), 1.5, 1.5, 0)
            lr_image = down_sample_image(blurred_image, reduce_factor)
            hr_batch.append(cropped_image)
            lr_batch.append(lr_image)
            plt.imshow(lr_image)
            plt.show()
            plt.imshow(cropped_image)
            plt.show()
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

def create_batches_input(data_batches, flow_path_batches):
    data_tuples = []
    for batch_idx in range(len(data_batches)):
        data_tuple = []
        for frame_idx in range(len(data_batches[batch_idx])):
            hr_frame = data_batches[batch_idx][frame_idx]
            flow_path = flow_path_batches[batch_idx][frame_idx]
            input = CellInput(flow_path, hr_frame)
            data_tuple.append(input)
        data_tuples.append(data_tuple)
    return data_tuples

def store_to_file(data_tuples, dataset):
    for data_tuple in data_tuples:
        dataset.append(data_tuple)

def calculate_flow(low_batch, flow_net):
    flow_results = []
    for idx in range(len(low_batch) - 1):
        flow_results.append(flow_net.inference_imgs(low_batch[idx], low_batch[idx + 1]))
    return flow_results

def calculate_batch_flows(lr_batches, flow_net):
    flow_batches = []
    for lr_batch in lr_batches:
        flow_batches.append(calculate_flow(lr_batch, flow_net))
    return flow_batches

def tranform_video(source_video_path, dest_video_path, flow_net):
    dataset = h5py.File(dest_video_path, 'w')
    reader = VideoReader(source_video_path)
    frames = reader.read_batch(10)
    i = 0
    while (len(frames) > 0):
        down_scaled = down_scale_batch(frames, 2)
        hr_batches, lr_batches = generate_random_batches(down_scaled, 256, 256, 1, 4)
        print(i)
        i += 1
        # call for the optical flow
        flow_path_batches = calculate_batch_flows(lr_batches, flow_net)
        create_batches_input(hr_batches, flow_path_batches)
        store_to_file(hr_batches, flow_path_batches)
        frames = reader.read_batch(10)
    return
