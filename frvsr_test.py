import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import frvsr.FRVSR_models as FRVSR_models

def trunc(tensor):
    # tensor = tensor.clone()
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    return tensor

def clip_output_img(hr_out):
    hr_out = trunc(hr_out.clone())
    hr_out = hr_out.cpu()
    out_img = hr_out.data[0].numpy()
    out_img *= 255.0
    return (np.uint8(out_img)).transpose((1, 2, 0))


if __name__ == "__main__":
    with torch.no_grad():
        parser = argparse.ArgumentParser(description='Test Single Video')
        parser.add_argument('--video', type=str, help='test low resolution video name')
        parser.add_argument('--model', type=str, help='generator model epoch name')
        parser.add_argument('--output', type=str, help='output video path')

        opt = parser.parse_args()

        UPSCALE_FACTOR = 4
        VIDEO_NAME = opt.video
        MODEL_NAME = opt.model
        OUTPUT_VIDEO = opt.output

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        videoCapture = cv2.VideoCapture(VIDEO_NAME)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        lr_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        lr_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sr_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
                         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)

        sr_video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, sr_video_size)

        success, frame = videoCapture.read()
        test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')

        model = FRVSR_models.FRVSR(0, 0, 0)
        model.to(device)
        model.load_state_dict(torch.load(MODEL_NAME, device))
        model.eval()
        model.set_param(batch_size=1, width=lr_width, height=lr_height)
        model.init_hidden(device)

        for index in test_bar:
            if success:
                image = Variable(ToTensor()(frame)).unsqueeze(0)
                image.to(device)
                if torch.cuda.is_available():
                    image = image.cuda()
                hr_out, lr_out = model(image)
                sr_video_writer.write(clip_output_img(hr_out))
                success, frame = videoCapture.read()
        sr_video_writer.release()
