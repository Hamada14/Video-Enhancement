from easydict import EasyDict as edict
import json

config = edict()

config.TRAIN = edict()

config.Model = edict()
##Model Parameters
config.TRAIN.time_steps = 10
config.Model.check_point_path = '/home/hamada/Video-Enhancement/srgans/check_points'
config.Model.high_width = 256
config.Model.high_height = 256
config.Model.low_width = 64
config.Model.low_height = 64

## Adam
config.TRAIN.data_points = 300,000
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 2
config.TRAIN.lr_decay_init = 0.1
config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 4
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.videos_path = '/home/hamada/Video-Enhancement/data_set'
config.TRAIN.flow_net_path = ''
config.VALID = edict()
## test set location
config.VALID.videos_path ='/home/hamada/Video-Enhancement/data_set'
config.VALID.flow_net_path = ''


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
