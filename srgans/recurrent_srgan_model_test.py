import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import scipy

import math
import tensorflow as tf
import tensorlayer as tl
import logging
from srgans.srgan_cell_model import *
from srgans.config import config, log_config
from srgans.lpipsMetric.lpips_tf import *

# you can configure the model hyper parameters by editing config.py
class RecurrentSRGAN():

    def __init__(self, high_width = config.Model.high_width, high_height = config.Model.high_height, low_width = config.Model.low_width,
       low_height = config.Model.low_height, batch_size = config.TRAIN.batch_size, time_steps = config.TRAIN.time_steps):

            self.batch_size = batch_size
            self.lr_init = config.TRAIN.lr_init
            self.beta1 = config.TRAIN.beta1
            self.time_steps = time_steps
            self.ni = int(np.sqrt(self.batch_size))
            self.high_width = high_width
            self.high_height = high_height
            self.low_width = low_width
            self.low_height = low_height
            tl.global_flag['mode'] = 'srgan'
            self.save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])

            logging.info("starting define model")

            #input LR batch image placeholder
            self.t_image = tf.placeholder('float32', [self.batch_size, self.time_steps, self.low_height, self.low_width,3],
                                          name='t_image_input_to_SRGAN_generator')
            # input HR batch image placeholder
            self.t_target_image = tf.placeholder('float32', [self.batch_size, self.time_steps, self.high_height, self.high_width, 3],
                                                 name='t_target_image')
            self.output_images = tf.Variable('float32', [self.batch_size, self.time_steps, self.high_height, self.high_width, 3],
                                                 name='output_image')
            #previous output frame after wrapping with optical flow
            # self.initializer = tf.constant(0.0 ,shape = [self.batch_size, self.high_width, self.high_height, 3])
            self.initial_estimate = tf.placeholder('float32',[self.batch_size, self.high_height, self.high_width, 3], name= 'initial_estimate')
            self.raw_optical_flow = tf.placeholder('float32',[self.batch_size, self.time_steps, self.low_height,self.low_width, 2]
                                            , name='raw_optical_flow')

            self.unrolled_d_total_loss = tf.Variable(0.0, name="D_unrolled_loss")
            self.unrolled_gan_total_loss = tf.Variable(0.0, name="GAN_unrolled_loss")
            self.unrolled_g_total_loss = tf.Variable(0.0, name="G_unrolled_loss")
            self.unrolled_mse_total_loss = tf.Variable(0.0, name="mse_unrolled_loss")
            self.unrolled_tloss_total_loss = tf.Variable(0.0, name="t_unrolled_loss")
            self.output_list = []

            #Unrolling the GAN model for t steps
            for t in range(self.time_steps):
                self.t_optical_flow = tf.image.resize_bilinear(tf.reshape(
                tf.slice(self.raw_optical_flow, [0, t, 0, 0, 0], [self.batch_size, 1, -1, -1, -1]), [self.batch_size,  self.low_height, self.low_width,2])
                , tf.constant([self.high_height, self.high_width]))
                if (t == 0):
                    self.t_wrapped_image = tf.contrib.image.dense_image_warp(self.initial_estimate, self.t_optical_flow)
                else:

                    self.t_wrapped_image = tf.contrib.image.dense_image_warp(self.output_image, self.t_optical_flow)
                time_step_target_image = tf.reshape(tf.slice(self.t_target_image, [0, t, 0, 0, 0], [self.batch_size, 1, -1, -1, -1]),[self.batch_size, self.high_height, self.high_width, 3])
                time_step_image = tf.reshape(tf.slice(self.t_image, [0, t, 0, 0, 0], [self.batch_size, 1, -1, -1, -1]),[self.batch_size, self.low_height, self.low_width, 3])
                self.net_g , self.output_image = SRGAN_generator(time_step_image, self.t_wrapped_image ,reuse=tf.AUTO_REUSE)

                # self.print_estimate = tf.print(self.output_image, [self.output_image])
                self.output_list.append(self.output_image)
            self.output_images = tf.stack(self.output_list)

            tl.global_flag['mode'] = 'srgan'
            self.save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
            self.save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
            tl.files.exists_or_mkdir(self.save_dir_ginit)
            tl.files.exists_or_mkdir(self.save_dir_gan)
            self.checkpoint_dir = config.Model.check_point_path
            tl.files.exists_or_mkdir(self.checkpoint_dir)

            #restore model
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            tl.layers.initialize_global_variables(self.sess)
            check_point = tf.train.get_checkpoint_state(self.checkpoint_dir)
            saver = tf.train.Saver(max_to_keep=2)
            if check_point and check_point.model_checkpoint_path:
                logging.info('Restored to a checkpoint stored at {}'.format(check_point.model_checkpoint_path))
                saver.restore(self.sess, check_point.model_checkpoint_path)
                global_step = int(check_point.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                logging.info('No checkpoint is found for SRGAN to load')

    def train_initial_generator(self, video_set):
        lr_test, hr_test, flow_test = self.sample_batch_for_test(video_set)
        ###========================== RESTORE MODEL =============================###
        configure = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configure.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
        sess = tf.Session(config=configure)
        tl.layers.initialize_global_variables(sess)
#        if tl.files.load_and_assign_npz(sess=sess, name=self.checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
#                                        network=self.net_g) is False:
#            tl.files.load_and_assign_npz(sess=sess,
#                                         name=self.checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
#                                         network=self.net_g)
        ###========================= initialize G ====================###

        ## initialize G
        n_epoch_init = config.TRAIN.n_epoch_init
        for epoch in range(0, n_epoch_init + 1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0
            steps = config.TRAIN.data_points // self.batch_size
            for idx in range(steps):
                step_time = time.time()
                lr_frame_input, hr_frame_input, flow_input = video_set.next_data()
                 #initial_output = sess.run(self.output_image)
                # writer = tf.summary.FileWriter("output", sess.graph)
                # sess.run([self.t_wrapped_image],{self.raw_optical_flow: flow_input,self.output_image: initial_output})
                # initial_output = sess.run(self.initial_output_image)
                #writer = tf.summary.FileWriter("output", sess.graph)

                errM,_ = sess.run([self.unrolled_mse_total_loss, self.g_init_train],
                                  feed_dict = {self.t_image: lr_frame_input, self.t_target_image: hr_frame_input
                                      , self.raw_optical_flow: flow_input})
                #writer.close()
                logging.info("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
                logging.info("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                    epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter))
            ## quick evaluation on train set
                if (idx % 50 == 0):
                    out = sess.run( self.output_images, {self.t_image: lr_test, self.t_target_image: hr_test, self.raw_optical_flow: flow_test})
                    print("[*] save images")#last time step of each batch
                    tl.vis.save_images(out[-1], [self.ni, self.ni], self.save_dir_ginit + '/train_epoch_%d_step_%d_.png' % (epoch, idx))
                    #lpips metric
                    #normalized [0,1]
                    out = (out - np.min(out)) / np.ptp(out)
                    lpips_dist = self.evaluate_with_lpips_metric(out,hr_test)
                    logging.info("Lpips Metric %s" % (str(lpips_dist)[1:-1]))
                 ## save model
                if (idx != 0) and (idx % 50 == 0):
                     tl.files.save_npz(self.net_g.all_params,
                                       name=self.checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    def train(self, video_set):

        global_step = -1
        f_lpips = open("/home/hamada/Video-Enhancement/srgans/lpips.txt", "a")
        f_loss = open("/home/hamada/Video-Enhancement/srgans/loss.txt", "a")

        lr_test, hr_test, flow_test = self.sample_batch_for_test(video_set)
        ###========================== RESTORE MODEL =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        check_point = tf.train.get_checkpoint_state(self.checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=2)
        if check_point and check_point.model_checkpoint_path:
                logging.info('Restored to a checkpoint stored at {}'.format(check_point.model_checkpoint_path))
                saver.restore(sess, check_point.model_checkpoint_path)
                global_step = int(check_point.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
                logging.info('No checkpoint is found for SRGAN to load')

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
                    global_step = global_step + 1
                    step_time = time.time()
                    lr_batches, hr_batches, flow_input = video_set.next_data()
                    ## update D
                    errD, _ = sess.run([self.unrolled_d_total_loss , self.d_train], {self.t_image: lr_batches,
                                                                                     self.t_target_image: hr_batches
                        ,self.raw_optical_flow : flow_input})
                    #update G
                    errC,errG, errM,errT,_ = sess.run([self.unrolled_g_total_loss, self.unrolled_gan_total_loss,
                                             self.unrolled_mse_total_loss,self.unrolled_tloss_total_loss, self.g_train],
                                             {self.t_image: lr_batches, self.t_target_image: hr_batches
                                                                                 ,self.raw_optical_flow : flow_input})
                    logging.info("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f)  (t_loss:%.6f)" %
                          (epoch, n_epoch, n_iter, time.time() - step_time, errD/self.time_steps, errG/self.time_steps,
                             errM/self.time_steps, errT /self.time_steps))

                    f_loss.write("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f)  (t_loss:%.6f) \n" %
                          (epoch, n_epoch, n_iter, time.time() - step_time, errD/self.time_steps, errG/self.time_steps,
                             errM/self.time_steps, errT /self.time_steps))

                    total_d_loss += errD
                    total_g_loss += errG
                    n_iter += 1

                    logging.info("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                        epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                        total_g_loss / n_iter))

                    f_loss.write("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f \n" % (
                        epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                        total_g_loss / n_iter))

                    if (idx % 50 == 0):
                        out = sess.run( self.output_images, {self.t_image: lr_test, self.t_target_image: hr_test, self.raw_optical_flow: flow_test})
                        print("[*] save images")#last step of each batch
                        tl.vis.save_images(out[-1], [self.ni, self.ni], self.save_dir_gan + '/train_epoch_%d_step_%d_.png' % (epoch, idx))
                        # lpips metric
                        # normalized [0,1]
                        out = (out - np.min(out)) / np.ptp(out)
                        lpips_dist = self.evaluate_with_lpips_metric(out, hr_test)
                        logging.info("Lpips Metric %s" % (str(lpips_dist)[1:-1]))
                        f_lpips.write ("Lpips Metric %s \n" % (str(lpips_dist)[1:-1]))
                        #save model
                    if (idx != 0) and (idx % 50 == 0):
                       saver.save(sess, self.checkpoint_dir + 'srgan.ckpt', global_step=global_step)

        f_lpips.close()
        f_loss.close()

    def sample_batch_for_test(self, video_set):

      ## use first `batch_size` of train set to have a quick test during training
      lr_frame_input, hr_frame_input, flow_input = video_set.next_data()
      tl.vis.save_images(np.array(lr_frame_input)[:,-1],[self.ni, self.ni], self.save_dir_ginit + '/_train_sample_96.png')
      tl.vis.save_images(np.array(hr_frame_input)[:,-1], [self.ni, self.ni], self.save_dir_ginit + '/_train_sample_384.png')
      tl.vis.save_images(np.array(lr_frame_input)[:,-1], [self.ni, self.ni], self.save_dir_gan + '/_train_sample_96.png')
      tl.vis.save_images(np.array(hr_frame_input)[:,-1], [self.ni, self.ni], self.save_dir_gan + '/_train_sample_384.png')

      return lr_frame_input, hr_frame_input,flow_input

    def estimate_frames (self, initializer, lr_input, flow_input):
      estimated = self.sess.run( self.output_images, {self.t_image: lr_input, self.raw_optical_flow: flow_input, self.initial_estimate: initializer})
      return estimated, lr_input
	
    def evaluate_with_lpips_metric(self, estimated_hr, hr):
        session = tf.Session()
        image0_ph = tf.placeholder(tf.float32)
        image1_ph = tf.placeholder(tf.float32)
        lpips_fn = session.make_callable(lpips(image0_ph,image1_ph),[image0_ph,image1_ph])
        distance = lpips_fn(hr, estimated_hr)
        return distance

    def evaluate_with_psnr_metric(self, estimated_hr, hr):
        mse = np.mean((estimated_hr - hr) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
