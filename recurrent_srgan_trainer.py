import logging
from logging.config import fileConfig

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from srgans.recurrent_srgan_model import RecurrentSRGAN
from srgans.config import config, log_config

fileConfig('logging_config.ini')

BATCH_SIZE = 8
FRAMES_LEN = 10
FRAME_TRY = 3

LR_HEIGHT = 64
LR_WIDTH = 64



video_dataset = VideoDataSet(
    FlowModelWrapper.getInstance(),
    config.TRAIN.videos_path,
    BATCH_SIZE,
    FRAMES_LEN,
    FRAME_TRY
)

srgan_model = RecurrentSRGAN()
segan_model = train_initial_generator(video_set)
srgan_model.train(video_dataset)
