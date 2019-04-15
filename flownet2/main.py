#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import os
import sys
import subprocess
import setproctitle
import colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

from . import models
from . import losses
from . import datasets
from .utils import flow_utils, tools

from .main_util import train, inference, get_default_argument_parser


# fp32 copy of parameters for update
global param_copy

def build_model(model='FlowNet2', check_point=None, loss_func='L1Loss'):
    parser = get_default_argument_parser()
    tools.add_arguments_for_module(
        parser, models, argument_for_class='model', default=model)

    tools.add_arguments_for_module(
        parser, losses, argument_for_class='loss', default=loss_func)

    tools.add_arguments_for_module(
        parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])

    tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training'})

    tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                       'replicates': 1})

    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                       'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[
            args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = torch.cuda.is_available()

        if args.inference:
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data, target, inference=True):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference:
                    return loss_values
                else:
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

        block.log('Number of parameters: {}'.format(sum([p.data.nelement(
        ) if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach()
                          for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if check_point and os.path.isfile(check_point):
            block.log("Loading checkpoint '{}'".format(check_point))
            checkpoint = torch.load(check_point)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(
                checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}'".format(check_point))

        else:
            block.log("No checkpoint found at '{}'".format(args.resume))
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments
    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        kwargs = tools.kwargs_from_args(args, 'optimizer')
        if args.fp16:
            optimizer = args.optimizer_class(
                [p for p in param_copy if p.requires_grad], **kwargs)
        else:
            optimizer = args.optimizer_class(
                [p for p in model_and_loss.parameters() if p.requires_grad], **kwargs)
        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    return model_and_loss
    # inference(args=args, data_loader=inference_loader, model=model_and_loss)
