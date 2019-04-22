import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

import os
from os.path import *
from glob import glob

from flownet2.main import build_model
from flownet2.utils import flow_utils, tools
from flownet2.main_util import train, inference, get_default_argument_parser
from flownet2.datasets import ImagesFromFolder
from flownet2 import datasets
from flownet2.utils import frame_utils as frame_utils
from flownet2.datasets import StaticCenterCrop, ImagesLoader

class FlowModelWrapper:
    __instance = None
    NUM_OF_WORKERS = 8
    BATCH_SIZE = 8
    DEFAULT_MODEL = 'FlowNet2'
    DEFAULT_CHECK_POINT = join(
        os.path.dirname(os.path.realpath(__file__)),
        'flownet2',
        'models',
        'FlowNet2_checkpoint.pth.tar'
    )

    @staticmethod
    def getInstance():
        """ Static access method. """
        if FlowModelWrapper.__instance == None:
            FlowModelWrapper()
        return FlowModelWrapper.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if FlowModelWrapper.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            FlowModelWrapper.__instance = self
            self.model = build_model(model=FlowModelWrapper.DEFAULT_MODEL,
                            check_point = FlowModelWrapper.DEFAULT_CHECK_POINT
                        )

    def inference_dir(self, src_dir, flow_dir, iext='ppm'):
        self.model.eval()
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if not os.path.isabs(src_dir):
            print("Relative")
            src_dir = join(dir_path, src_dir)
        if not os.path.isabs(flow_dir):
            flow_dir = join(dir_path, flow_dir)

        inference_size = [-1, -1]

        data_loader = DataLoader(
            ImagesFromFolder(inference_size, root=src_dir, iext=iext),
            batch_size=FlowModelWrapper.BATCH_SIZE, shuffle=False, num_workers=FlowModelWrapper.NUM_OF_WORKERS)
        progress = tqdm(data_loader, ncols=100, total=len(data_loader), desc='Inferencing ',
                        leave=True)

        for batch_idx, (data, target) in enumerate(progress):
            data, target = [d.cuda(non_blocking=True) for d in data], [
                t.cuda(non_blocking=True) for t in target]
            data, target = [Variable(d) for d in data], [
                Variable(t) for t in target]

            with torch.no_grad():
                losses, output = self.model(data[0], target[0], inference=True)

            for i in range(len(output)):
                _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow(join(flow_dir, '%07d.flo' %
                                          (batch_idx * FlowModelWrapper.BATCH_SIZE + i)), _pflow)
            progress.update(1)


    def inference_imgs(self, img1, img2):
        self.model.eval()
        images = np.array([img1, img2]).transpose(3, 0, 1, 2)
        data = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        target = torch.zeros(data.size()[0:1] + (2,) + data.size()[-2:])
        data, target = [Variable(data.cuda(non_blocking=True))], [Variable(target.cuda(non_blocking=True))]
        with torch.no_grad():
            losses, result = self.model(data[0], target[0], inference=True)
        return result[0].data.cpu().numpy().transpose(1, 2, 0)



def dir_example():
    s = FlowModelWrapper.getInstance()
    s.inference_dir('flownet2/examples', 'flownet2/examples')

def imgs_example():
    s = FlowModelWrapper.getInstance()
    src_path = join(
        os.path.dirname(os.path.realpath(__file__)),
        'flownet2',
        'examples',
        '*.ppm'
    )
    images_dir = sorted(glob(src_path))
    images = [
        frame_utils.read_gen(images_dir[0]),
        frame_utils.read_gen(images_dir[1])
    ]
    flow = s.inference_imgs(images[0], images[1])

    flow_path = join(
        os.path.dirname(os.path.realpath(__file__)),
        'flownet2',
        'examples',
        '0001.flo'
    )
    flow_utils.writeFlow(flow_path, flow)
