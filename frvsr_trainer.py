import logging
from logging.config import fileConfig

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from frvsr.model import FRVSR

fileConfig('logging_config.ini')

BATCH_SIZE = 4
FRAMES_LEN = 10
FRAME_TRY = 3

LR_HEIGHT = 64
LR_WIDTH = 64
IMAGE_CHANNELS = 3
FLOW_DEPTH = 2
CHECK_POINT_PATH = '/home/moamen/gp/Video-Enhancement/check_point/frvsr'

video_dataset = VideoDataSet(
    FlowModelWrapper.getInstance(),
    '/home/hamada/dataset/backup',
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
