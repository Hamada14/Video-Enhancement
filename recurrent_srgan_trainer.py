import logging
from logging.config import fileConfig

from flow_model_wrapper import FlowModelWrapper
from pre_processing.video_dataset import VideoDataSet
from srgans.recurrent_srgan_model import RecurrentSRGAN
from srgans.config import config, log_config



BATCH_SIZE = config.TRAIN.batch_size
FRAMES_LEN = config.TRAIN.time_steps
FRAME_TRY = 3



HIGH_IMG_SIZE = 256
SCALE_FACTOR = 4

video_dataset = VideoDataSet(
    FlowModelWrapper.getInstance(),
    config.TRAIN.videos_path,
    BATCH_SIZE,
    FRAMES_LEN,
    FRAME_TRY,
    HIGH_IMG_SIZE,
    SCALE_FACTOR
)

srgan_model = RecurrentSRGAN()
for i in range(10):
    srgan_model.sample_batch_for_test(video_dataset)
srgan_model.train_initial_generator(video_dataset)
srgan_model.train(video_dataset)
