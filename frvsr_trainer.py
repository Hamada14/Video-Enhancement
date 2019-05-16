import logging
import os
from logging.config import fileConfig
from datetime import datetime

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from frvsr.model import FRVSR


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

logger = logging.getLogger()

CHECK_POINT_PATH = os.path.join(dir_path, 'check_point/frvsr/')
DATA_SET_PATH = os.path.join(dir_path, 'data_set')


def build_validation_data():
    data = []
    videos_count = 10
    snapshot_count = 10
    skip_size = 10

    BATCH_SIZE = 4
    FRAMES_LEN = 10
    FRAME_TRY = 1

    logger.debug('Building the validation dataset')

    video_dataset = VideoDataSet(
        FlowModelWrapper.getInstance(),
        DATA_SET_PATH,
        BATCH_SIZE,
        FRAMES_LEN,
        FRAME_TRY,
        HIGH_IMG_SIZE,
        SCALE_FACTOR
    )

    for video_idx in range(videos_count):
        for snapshot in range(snapshot_count):
            for skip in range(skip_size):
                video_dataset.skip_data()
            data.append(video_dataset.next_data())
        video_dataset.skip_video()

    logger.debug('Finished building the validation dataset')
    return data


def train():
    BATCH_SIZE = 4
    FRAMES_LEN = 10
    FRAME_TRY = 10

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

    frvsr_model.train(video_dataset, build_validation_data())


def inference():
    BATCH_SIZE = 1 # change to 4
    FRAMES_LEN = 4 # change to 10
    FRAME_TRY = 1 # change to 15

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

    for i in range(10):
        lr, hr, flow = video_dataset.next_data()
        frvsr_model.test_inference(lr, flow, hr)
train()
#inference()
