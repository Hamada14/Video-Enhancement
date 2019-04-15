from flow_model_wrapper import FlowModelWrapper
from frvsr.pre_processing.video_transformation import tranform_video

src_video = '/home/hamada/dataset/backup/Bella Italia pt2 4K-174952003.mp4'
dest_dataset = 'd1.hdf5'

flow_net = FlowModelWrapper.getInstance()

tranform_video(src_video, dest_dataset, flow_net)
