import logging
from logging.config import fileConfig

from flow_model_wrapper import FlowModelWrapper
from frvsr.pre_processing.video_transformation import tranform_video

fileConfig('logging_config.ini')


src_video = '/home/hamada/dataset/backup/Bella Italia pt2 4K-174952003.mp4'

flow_net = FlowModelWrapper.getInstance()

tranform_video(src_video, flow_net)
