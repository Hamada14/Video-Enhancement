import sys

from flow_model_wrapper import FlowModelWrapper
from frvsr.pre_processing.video_transformation import tranform_video

src_video = sys.argv[1]
dest_dataset = sys.argv[2]

flow_net = FlowModelWrapper.getInstance()

transform(src_video, dest_dataset, flow_net)
