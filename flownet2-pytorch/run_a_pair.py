import torch
import numpy as np
import argparse


from utils.frame_utils import read_gen#the path is depended on where you create this module
from networks.FlowNetS import FlowNetS#the path is depended on where you create this module

model = "models/FlowNet2-S_checkpoint.pth.tar"
im1 = "examples/0000000-img0.ppm"
im2 = "examples/0000000-img1.ppm"
out_flow = "examples/0000000-img.flo"

if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    #initial a Net
    net = FlowNetS(args).cuda()
    #load the state_dict
    dict = torch.load(model)
    net.load_state_dict(dict["state_dict"])

    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(im1)
    pim2 = read_gen(im2)
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    #process the image pair to obtian the flow
    result = net(im).squeeze()


    #save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow(out_flow,data)
