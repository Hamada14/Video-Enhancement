from datetime import datetime
from pre_processing.video_dataset import VideoDataSet
from torch.autograd import Variable
import frvsr.FRVSR_models as FRVSR_models
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import gc
from tqdm import tqdm
from tqdm import trange


torch.backends.cudnn.benchmark = True


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


def load_model(path, batch_size, width, height):
    logger.debug('Loading the FRVSR model')
    model = FRVSR_models.FRVSR(batch_size=batch_size, lr_height=height, lr_width=width)
    if os.path.isfile(path):
        logger.debug('No previous checkpoint found')
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)
    return model

def train(model, device, data_set, checkpoint_path):
    num_epochs = 25
    content_criterion = FRVSR_models.Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epoch = 1
    print('Starting training')
    while epoch <= num_epochs:
        train_loss = 0.0
        model.train()
        steps = 1000
        progress_bar = trange(steps, desc='Training', leave=True)
        for j in progress_bar:
            lr_imgs, hr_imgs = data_set.next_data()
            lr_imgs = torch.stack(lr_imgs, dim=0)
            hr_imgs = torch.stack(hr_imgs, dim=0)
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            optimizer.zero_grad()
            model.init_hidden(device)
            batch_content_loss = 0
            batch_flow_loss = 0
            cnt = 0
            for lr_img, hr_img in zip(lr_imgs, hr_imgs):
                hr_est, lr_est = model(lr_img)
                content_loss = content_criterion(hr_est, hr_img)
                flow_loss = torch.mean((lr_img - lr_est) ** 2)
                batch_content_loss += content_loss
                if cnt > 0:
                    batch_flow_loss += flow_loss
                cnt += 1
            loss = batch_content_loss + batch_flow_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description('last loss : {:.8f}, average loss: {:.8f}'.format(loss.item(), train_loss/(j + 1)))
            progress_bar.refresh()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), checkpoint_path)
        gc.collect()


        epoch += 1

def validate(model, device):
    model.eval()
    with torch.no_grad():
        output_period = 0
        running_loss = 0
        for batch_num, (lr_imgs, hr_imgs) in enumerate(val_loader, 1):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            model.init_hidden(device)
            batch_content_loss = 0
            batch_flow_loss = 0

            # lr_imgs = 7 * 4 * 3 * H * W
            cnt = 0
            for lr_img, hr_img in zip(lr_imgs, hr_imgs):
                # print(lr_img.shape)
                hr_est, lr_est = model(lr_img)
                content_loss = content_criterion(hr_est, hr_img)
                flow_loss = torch.mean((lr_img - lr_est) ** 2)
                # flow_loss = ssim_loss(lr_img, lr_est)
                # print(f'content_loss is {content_loss}, flow_loss is {flow_loss}')
                batch_content_loss += content_loss
                if cnt > 0:
                    batch_flow_loss += flow_loss
                cnt += 1
            output_period += 1
            loss = batch_content_loss + batch_flow_loss
            running_loss += loss
            epoch_valid_loss = (epoch_valid_loss * (batch_num - 1) + loss) / batch_num



FRAMES_LEN = 10
BATCH_SIZE = 4
width, height = 64, 64

HIGH_IMG_SIZE = 256
SCALE_FACTOR = 4
FRAME_TRY = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
CHECK_POINT_PATH = os.path.join(dir_path, 'check_point/frvsr')
DATA_SET_PATH = os.path.join(dir_path, 'data_set')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(CHECK_POINT_PATH, BATCH_SIZE, width, height)
model = model.to(device)

video_dataset = VideoDataSet(
    DATA_SET_PATH,
    BATCH_SIZE,
    FRAMES_LEN,
    FRAME_TRY,
    HIGH_IMG_SIZE,
    SCALE_FACTOR
)
train(model, device, video_dataset, CHECK_POINT_PATH)
