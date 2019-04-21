import logging
import os
from logging.config import fileConfig

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from frvsr.model import FRVSR

fileConfig('logging_config.ini')

BATCH_SIZE = 1
FRAMES_LEN = 4
FRAME_TRY = 3

LR_HEIGHT = 64
LR_WIDTH = 64
IMAGE_CHANNELS = 3
FLOW_DEPTH = 2


dir_path = os.path.dirname(os.path.realpath(__file__))


CHECK_POINT_PATH = os.path.join(dir_path, 'check_point/frvsr/')
DATA_SET_PATH = os.path.join(dir_path, 'data_set')

video_dataset = VideoDataSet(
    FlowModelWrapper.getInstance(),
    DATA_SET_PATH,
    BATCH_SIZE,
    FRAMES_LEN,
    FRAME_TRY
)

frvsr_model = FRVSR(
    BATCH_SIZE,
    FRAMES_LEN,
    LR_HEIGHT,
    LR_WIDTH,
    IMAGE_CHANNELS,
    FLOW_DEPTH,
    CHECK_POINT_PATH
)

frvsr_model.train(video_dataset)
