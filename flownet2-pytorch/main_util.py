import torch
import os
import argparse
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from os.path import *

from utils import flow_utils, tools

# Reusable function for training and validataion


def train(args, epoch, start_iteration, data_loader, model, optimizer, logger, is_validate=False, offset=0):
    statistics = []
    total_loss = 0

    if is_validate:
        model.eval()
        title = 'Validating Epoch {}'.format(epoch)
        args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(
            data_loader), args.validation_n_batches), leave=True, position=offset, desc=title)
    else:
        model.train()
        title = 'Training Epoch {}'.format(epoch)
        args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(
            data_loader), args.train_n_batches), smoothing=.9, miniters=1, leave=True, position=offset, desc=title)

    last_log_time = progress._time()
    for batch_idx, (data, target) in enumerate(progress):

        data, target = [Variable(d) for d in data], [
            Variable(t) for t in target]
        if args.cuda and args.number_gpus == 1:
            data = [d.cuda(non_blocking=True) for d in data]
            target = [t.cuda(non_blocking=True) for t in target]

        optimizer.zero_grad() if not is_validate else None
        losses = model(data[0], target[0])
        losses = [torch.mean(loss_value) for loss_value in losses]
        loss_val = losses[0]  # Collect first loss for weight update
        total_loss += loss_val.data[0]
        loss_values = [v.data[0] for v in losses]

        # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
        loss_labels = list(model.module.loss.loss_labels)

        assert not np.isnan(total_loss)

        if not is_validate and args.fp16:
            loss_val.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm(
                    model.parameters(), args.gradient_clip)

            params = list(model.parameters())
            for i in range(len(params)):
                param_copy[i].grad = params[i].grad.clone().type_as(
                    params[i]).detach()
                param_copy[i].grad.mul_(1. / args.loss_scale)
            optimizer.step()
            for i in range(len(params)):
                params[i].data.copy_(param_copy[i].data)

        elif not is_validate:
            loss_val.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm(
                    model.parameters(), args.gradient_clip)
            optimizer.step()

        # Update hyperparameters if needed
        global_iteration = start_iteration + batch_idx
        if not is_validate:
            tools.update_hyperparameter_schedule(
                args, epoch, global_iteration, optimizer)
            loss_labels.append('lr')
            loss_values.append(optimizer.param_groups[0]['lr'])

        loss_labels.append('load')
        loss_values.append(progress.iterable.last_duration)

        # Print out statistics
        statistics.append(loss_values)
        title = '{} Epoch {}'.format(
            'Validating' if is_validate else 'Training', epoch)

        progress.set_description(
            title + ' ' + tools.format_dictionary_of_losses(loss_labels, statistics[-1]))

        if ((((global_iteration + 1) % args.log_frequency) == 0 and not is_validate) or
                (is_validate and batch_idx == args.validation_n_batches - 1)):

            global_iteration = global_iteration if not is_validate else start_iteration

            logger.add_scalar('batch logs per second', len(
                statistics) / (progress._time() - last_log_time), global_iteration)
            last_log_time = progress._time()

            all_losses = np.array(statistics)

            for i, key in enumerate(loss_labels):
                logger.add_scalar('average batch ' + str(key),
                                  all_losses[:, i].mean(), global_iteration)
                logger.add_histogram(
                    str(key), all_losses[:, i], global_iteration)

        # Reset Summary
        statistics = []

        if (is_validate and (batch_idx == args.validation_n_batches)):
            break

        if ((not is_validate) and (batch_idx == (args.train_n_batches))):
            break

    progress.close()

    return total_loss / float(batch_idx + 1), (batch_idx + 1)


# Reusable function for inference
def inference(args, data_loader, model, offset=0):

    model.eval()

    if args.save_flow or args.render_validation:
        flow_folder = "{}/inference".format(args.save)
        if not os.path.exists(flow_folder):
            os.makedirs(flow_folder)

    args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

    progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ',
                    leave=True, position=offset)

    statistics = []
    total_loss = 0
    for batch_idx, (data, target) in enumerate(progress):
        if args.cuda:
            data, target = [d.cuda(non_blocking=True) for d in data], [
                t.cuda(non_blocking=True) for t in target]
        data, target = [Variable(d) for d in data], [
            Variable(t) for t in target]

        with torch.no_grad():
            losses, output = model(data[0], target[0], inference=True)

        # import IPython; IPython.embed()
        if args.save_flow or args.render_validation:
            for i in range(args.inference_batch_size):
                _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (
                    batch_idx * args.inference_batch_size + i)),  _pflow)

        progress.update(1)

        if batch_idx == (args.inference_n_batches - 1):
            break

    progress.close()

    return


def get_default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[
        256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument('--number_workers', '-nw',
                        '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int,
                        default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str,
                        help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work',
                        type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int,
                        default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+',
                        default=[-1, -1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true',
                        help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter',
                        type=int, default=1, help="Log every n batches")

    parser.add_argument('--fp16', action='store_true',
                        help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    return parser
