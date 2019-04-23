import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import scipy


import tensorflow as tf
import tensorlayer as tl
import logging
from srgans.srgan_cell_model import *
from srgans.config import config, log_config



# you can configure the model hyper parameters by editing config.py
class RecurrentSRGAN():

    def __init__(self):

        self.batch_size = config.TRAIN.batch_size
        self.lr_init = config.TRAIN.lr_init
        self.beta1 = config.TRAIN.beta1

        self.time_steps = config.TRAIN.time_steps
        self.high_width = config.Model.high_width
        self.high_height = config.Model.high_height
        self.low_width = config.Model.low_width
        self.low_height = config.Model.low_height
       



        logging.info("starting define model")


        #input LR batch image placeholder
        self.t_image = tf.placeholder('float32', [self.batch_size, self.time_steps, self.low_width, self.low_height, 3],
                                      name='t_image_input_to_SRGAN_generator')
        # input HR batch image placeholder
        self.t_target_image = tf.placeholder('float32', [self.batch_size, self.time_steps, self.high_width, self.high_height, 3],
                                             name='t_target_image')
        #previous output frame after wrapping with optical flow
        self.initial_output_image = self.output_image = tf.Variable(tf.zeros((self.batch_size, self.high_width, self.high_height, 3)), name="output_HR_frame")

        self.raw_optical_flow = tf.placeholder('float32',[self.batch_size, self.time_steps, self.low_width, self.low_height, 2]
                                        , name='t_optical_flow')
        self.t_optical_flow = tf.placeholder('float32',[self.batch_size, self.high_width, self.high_height, 2]
                                        , name='t_optical_flow')

        self.unrolled_d_total_loss = tf.Variable(0.0, name="D_unrolled_loss")
        self.unrolled_g_total_loss = tf.Variable(0.0, name="G_unrolled_loss")
        self.unrolled_mse_total_loss = tf.Variable(0.0, name="mse_unrolled_loss")

        #Unrolling the GAN model for t steps
        for t in range(self.time_steps):
	    #logging.info("t optical flows")
            #logging.info(self.t_optical_flow[:,t])
	    #logging.info("t self output image")
            #logging.info(self.t_optical_flow[:,t])

            self.t_optical_flow = tf.image.resize_bilinear(self.raw_optical_flow[:,t], tf.constant([256, 256]))
            self.t_wrapped_image = tf.contrib.image.dense_image_warp(self.output_image, self.t_optical_flow)
            self.net_g , self.output_image = SRGAN_generator(self.t_image[:,t], self.t_wrapped_image ,
                                                               reuse=tf.AUTO_REUSE)
            self.net_d, self.logits_real = SRGAN_discriminator(self.t_target_image[:,t],
                                                               is_train=True, reuse=tf.AUTO_REUSE)
            _, self.logits_fake = SRGAN_discriminator(self.output_image, is_train=True, reuse=tf.AUTO_REUSE)
            d_loss1 = tl.cost.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real), name='d1')
            d_loss2 = tl.cost.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake), name='d2')
            d_loss = d_loss1 + d_loss2
            self.unrolled_d_total_loss += d_loss
            # TODO check multiply here or at the end only multiply total loss???
            g_gan_loss = 3e-3 * tl.cost.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake), name='g')
            mse_loss = tl.cost.mean_squared_error(self.output_image,self.t_target_image[:,t], is_mean=True)
            self.unrolled_mse_total_loss += mse_loss
            g_loss = mse_loss + g_gan_loss
            self.unrolled_g_total_loss += g_loss
            self.net_g.print_params(False)
            self.net_d.print_params(False)

        self.g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
        self.d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

        
        with tf.variable_scope('learning_rate'):
            self.lr_v = tf.Variable(self.lr_init, trainable=False)
        ## Pretrain with mse first
        self.g_init_train = tf.train.AdamOptimizer(self.lr_v, beta1= self.beta1).minimize(self.unrolled_mse_total_loss, var_list=self.g_vars)

        ## SRGAN train with adversrial loss next
        
        self.g_train = tf.train.AdamOptimizer(self.lr_v, beta1= self.beta1).minimize(self.unrolled_g_total_loss, var_list=self.g_vars)
        self.d_train = tf.train.AdamOptimizer(self.lr_v, beta1= self.beta1).minimize(self.unrolled_d_total_loss, var_list= self.d_vars)

    def train_initial_generator(self, video_set):
        ## initialize G
        n_epoch_init = config.TRAIN.n_epoch_init
        ## create folders to save result images and trained model
        tl.global_flag['mode'] = 'srgan'
        save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
        tl.files.exists_or_mkdir(save_dir_gan)
        checkpoint_dir = config.Model.check_point_path
        tl.files.exists_or_mkdir(checkpoint_dir)
        ###========================== RESTORE MODEL =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                        network=self.net_g) is False:
            tl.files.load_and_assign_npz(sess=sess,
                                         name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                                         network=self.net_g)

        ###========================= initialize G ====================###


        for epoch in range(0, n_epoch_init + 1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0
            steps = config.TRAIN.data_points // self.batch_size
            for idx in range(steps):
                step_time = time.time()
                lr_frame_input, hr_frame_input, flow_input = video_set.next_data()         
                initial_output = sess.run(self.initial_output_image)
                errM, _ = sess.run([self.unrolled_mse_total_loss, self.g_init_train],
                                  {self.t_image: lr_frame_input, self.t_target_image: hr_frame_input
                                       , self.raw_optical_flow: flow_input,
                                     self.output_image: initial_output})
                logging.info("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
                logging.info("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                    epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter))

                ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(self.net_g.all_params,
                                  name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    def train(self, video_set):
        tl.global_flag['mode'] = 'srgan'
        save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
        tl.files.exists_or_mkdir(save_dir_gan)
        checkpoint_dir = config.Model.check_point_path
        tl.files.exists_or_mkdir(checkpoint_dir)
        ###========================== RESTORE MODEL =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                        network=self.net_g) is False:
            tl.files.load_and_assign_npz(sess=sess,
                                         name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                                         network=self.net_g)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
                                     network=self.net_d)

        ## adversarial learning (SRGAN)
        n_epoch = config.TRAIN.n_epoch
        lr_decay = config.TRAIN.lr_decay
        decay_every = config.TRAIN.decay_every


        ###========================= Training G ====================###
        ## fixed learning rate
        sess.run(tf.assign(self.lr_v, self.lr_init))
        logging.info(" ** fixed learning rate: %f (for init G)" % self.lr_init)
        for epoch in range(0, n_epoch + 1):
            ## update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(self.lr_v, self.lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (self.lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(self.lr_v, self.lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (self.lr_init, decay_every, lr_decay)
                logging.info(log)

            epoch_time = time.time()
            total_d_loss, total_g_loss, n_iter = 0, 0, 0
            steps = config.TRAIN.data_points // self.batch_size
            for idx in range(steps):
                    step_time = time.time()
                    lr_batches, hr_batches, flow_input = video_set.next_data()
                    initial_output = sess.run(self.initial_output_image)
                    ## update D
                    errD, _ = sess.run([self.unrolled_d_total_loss , self.d_train], {self.t_image: lr_batches,
                                                                                     self.t_target_image: hr_batches
                        ,self.raw_optical_flow : flow_input, self.output_image : initial_output})
                    #update G
                    errG, errM, _ = sess.run([self.unrolled_g_total_loss, self.unrolled_mse_total_loss, self.g_train],
                                             {self.t_image: lr_batches, self.t_target_image: hr_batches
                                                                                 ,self.raw_optical_flow : flow_input,
                                                                                 self.output_image : initial_output})
                    logging.info("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f)" %
                          (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM))
                    total_d_loss += errD
                    total_g_loss += errG
                    n_iter += 1

                    logging.info("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                        epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                        total_g_loss / n_iter))



        # save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(self.net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                  sess=sess)
                tl.files.save_npz(self.net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
                                  sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    ##load validation video list
    valid_video_list = sorted(tl.files.load_file_list(path=config.TRAIN.videos_path, regx='.*.mp4', printable=False))


