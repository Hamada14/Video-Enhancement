import logging
import os
from logging.config import fileConfig
from datetime import datetime

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from frvsr.model import FRVSR

BATCH_SIZE = 1
FRAMES_LEN = 4
FRAME_TRY = 3

LR_HEIGHT = 64
LR_WIDTH = 64
IMAGE_CHANNELS = 3
FLOW_DEPTH = 2
HIGH_IMG_SIZE = 256
SCALE_FACTOR = 4

dir_path = os.path.dirname(os.path.realpath(__file__))
now = datetime.now()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(os.path.join(dir_path, 'logs'), str('log-' +now.strftime("%m-%d-%Y-%H-%M-%S")))),
        logging.StreamHandler()
])

CHECK_POINT_PATH = os.path.join(dir_path, 'check_point/frvsr/')
DATA_SET_PATH = os.path.join(dir_path, 'data_set')

video_dataset = VideoDataSet(
    FlowModelWrapper.getInstance(),
    DATA_SET_PATH,
    BATCH_SIZE,
    FRAMES_LEN,
    FRAME_TRY,
    HIGH_IMG_SIZE,
    SCALE_FACTOR
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
