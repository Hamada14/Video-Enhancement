from flow_model_wrapper import FlowModelWrapper
from frvsr.pre_processing.video_transformation import tranform_video

src_video = '/home/moamen/gp/Video-Enhancement/dataset/4K Cinema Camera - With Voiceover-121649159.mp4'
dest_dataset = 'd1.hdf5'

flow_net = FlowModelWrapper.getInstance()

tranform_video(src_video, dest_dataset, flow_net)
