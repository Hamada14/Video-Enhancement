from video_reader import VideoReader
from cell_input import CellInput
import random
from skimage.transform import resize

def generate_random_batches(frames, new_width, new_height, number_of_batches):
    new_batches = []
    for batch_idx in range(number_of_batches):
        dimensions = frames[0].shape
        start_w = random.randint(0, dimensions[0] - new_width)
        start_h = random.randint(0, dimensions[1] - new_height)
        new_batch = []
        for frame_idx in range(len(frames)):
            cur_frame = frames[frame_idx]
            new_batch.append(cur_frame[start_w : start_w + new_width, start_h : start_h + new_height])
        new_batches.append(new_batch)
    return new_batches

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

def tranform_video(source_video_path, dest_video_path):
    reader = VideoReader(source_video_path)
    frames = reader.read_batch(10)

    while (len(frames) > 0):
        down_scaled = down_scale_batch(frames, 2)
        new_batches = generate_random_batches(down_scaled, 256, 256, 1)
        # call for the optical flow
        flow_path_batches = []
        # create_batches_input(new_batches, flow_path_batches)
        frames = reader.read_batch(10)
    return


tranform_video('/home/moamen/dataset/AJA Ki Pro Quad - Efficient 4K workflows.-40439273.mov', '')