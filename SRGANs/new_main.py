import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import logging

from frvsr.pre_processing.video_reader import VideoReader
import tensorflow as tf
import SRGANs.tensorlayer as tl
from SRGANs.model import *
from SRGANs.utils import *
from SRGANs.config import config, log_config
from flow_model_wrapper import FlowModelWrapper
from SRGANs.gan_video_transformation import *
###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
time_steps = config.TRAIN.time_steps

ni = int(np.sqrt(batch_size))

flow_estimator = FlowModelWrapper.getInstance()
logger = logging.getLogger()

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_video_list = sorted(tl.files.load_file_list(path=config.TRAIN.videos_path, regx='.*.mp4', printable=False))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, time_steps, 32, 32, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, time_steps, 128, 128, 3], name='t_target_image')
    initial_output_image = output_image = tf.Variable(tf.zeros((batch_size,128,128,3)), name="output frame")
    t_optical_flow = tf.placeholder('float32',[batch_size, time_steps, 128, 128, 3], name='t_optical_flow')
    #intialize output_image to be black image

    unrolled_d_total_loss = tf.Variable(tf.zeros((batch_size, 1)), name="D unrolled loss")
    unrolled_g_total_loss = tf.Variable(tf.zeros((batch_size, 1)), name="G unrolled loss")
    unrolled_mse_total_loss = tf.Variable(tf.zeros((batch_size, 1)), name="mse unrolled loss")
    for t in range(time_steps):
        t_wrapped_image = tf.contrib.image.dense_image_warp(output_image, t_optical_flow[:,t])
        net_g , output_image = SRGAN_generator(t_image[:,t], t_wrapped_image , is_train=True, reuse=tf.AUTO_REUSE)
        net_d, logits_real = SRGAN_discriminator(t_target_image[:,t], is_train=True, reuse=tf.AUTO_REUSE)
        _, logits_fake = SRGAN_discriminator(output_image, is_train=True, reuse=tf.AUTO_REUSE)
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        d_loss = d_loss1 + d_loss2
        unrolled_d_total_loss = tf.Variable.assign_add(d_loss)
        # TODO check multiply here or at the end only multiply total loss???
        g_gan_loss = 3e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
        mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
        unrolled_mse_total_loss = tf.Variable.assign_add(mse_loss)
        g_loss = mse_loss + g_gan_loss
        unrolled_g_total_loss = tf.Variable.assign_add(g_loss)
        net_g.print_params(False)
        net_d.print_params(False)

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Pretrain with mse first
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(unrolled_mse_total_loss, var_list=g_vars)

    ## SRGAN train with adversrial loss next
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(unrolled_g_total_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(unrolled_d_total_loss, var_list=d_vars)

###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

###========================= initialize G ====================###

    for epoch in range(0, n_epoch  + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        for idx in range(len(train_video_list)):
            logger.info('Transforming video {' + train_video_list[idx] + '}')
            logger.info('Successfully created the destination file')
            logger.info('Reading the video file')
            reader = VideoReader(train_video_list[idx])
            frames = reader.read_batch(10)
            i = 0
            while (len(frames) >= 10):
                step_time = time.time()
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Read the input batch successfully')
                down_scaled = down_scale_batch(frames, 2)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Downscaled the frames successfully')
                hr_batches, lr_batches = generate_random_batches(down_scaled, HIGH_WIDTH, HIGH_HEIGHT, BATCH_SIZE,
                                                                 HIGH_WIDTH // LOW_WIDTH)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Generated random batches from downscaled frames')
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Initializing flow calculation')
                flow_path_batches = calculate_batch_flows(lr_batches, flow_estimator)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Finished calculating the flow for the current batch')
                initial_output = initial_output_image.eval()
                errM, _ = sess.run([unrolled_mse_total_loss, g_optim_init], {t_image: lr_batches, t_target_image: hr_batches
                                                                             ,t_optical_flow : flow_path_batches,
                                                                             output_image : initial_output_image})
                logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
                logger.info("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter))
                # train the SR model
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Appended the batch to the dataset file')
                frames = reader.read_batch(10)
                i += 1
                ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params,
                                name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

###========================= Training G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    logger.info(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            logger.info(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx in range(len(train_video_list)):
            logger.info('Transforming video {' + train_video_list[idx] + '}')
            logger.info('Successfully created the destination file')
            logger.info('Reading the video file')
            reader = VideoReader(train_video_list[idx])
            frames = reader.read_batch(10)
            i = 0
            while (len(frames) >= 10):
                step_time = time.time()
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Read the input batch successfully')
                down_scaled = down_scale_batch(frames, 2)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Downscaled the frames successfully')
                hr_batches, lr_batches = generate_random_batches(down_scaled, HIGH_WIDTH, HIGH_HEIGHT, BATCH_SIZE,
                                                                 HIGH_WIDTH // LOW_WIDTH)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Generated random batches from downscaled frames')
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Initializing flow calculation')
                flow_path_batches = calculate_batch_flows(lr_batches, flow_estimator)
                logger.info('[batch: ' + str(i + 1) + '] ' + 'Finished calculating the flow for the current batch')
                initial_output = initial_output_image.eval()
                ## update D
                errD, _ = sess.run([unrolled_d_total_loss , d_optim], {t_image: lr_batches, t_target_image: hr_batches
                    ,t_optical_flow : flow_path_batches, output_image : initial_output_image})

                errG, errM, _ = sess.run([unrolled_g_total_loss, unrolled_mse_total_loss, g_optim], {t_image: lr_batches, t_target_image: hr_batches
                                                                             ,t_optical_flow : flow_path_batches,
                                                                             output_image : initial_output_image})
                logger.info("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f)" %
                      (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1

            logger.info("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
            epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
            total_g_loss / n_iter))

            logger.info('[batch: ' + str(i + 1) + '] ' + 'Appended the batch to the dataset file')
            frames = reader.read_batch(10)
            i += 1
            ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(net_g.all_params,
                                name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)



def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    ##load validation video list
    valid_video_list = sorted(tl.files.load_file_list(path=config.TRAIN.videos_path, regx='.*.mp4', printable=False))