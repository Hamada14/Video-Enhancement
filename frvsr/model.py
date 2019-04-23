import os
import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm
from tqdm import trange


from scipy import misc
from PIL import Image


logger = logging.getLogger()

HR_INDEX = 2
LR_INDEX = 0
FLOW_INDEX = 1

class FRVSR():
    def __init__(self, batch_size, frames_len, height, width, channels, flow_depth, check_point_path):

        lr_frame_input = tf.placeholder(dtype=tf.float64)
        flow_input = tf.placeholder(dtype=tf.float64)
        hr_frame_input = tf.placeholder(dtype=tf.float64)


        def step(previous_output, current_input):
            lr_frame, flow = tf.split(current_input, [3, 2], axis=-1)
            with tf.variable_scope("frvsr_model", reuse=tf.AUTO_REUSE):
                upscaled_flow = tf.image.resize_bilinear(flow, tf.constant([256, 256]))
                # warpping the last estimate with the flow
                warped_frame = tf.contrib.image.dense_image_warp(previous_output, upscaled_flow)
                # space to depth for warpped frame
                depth = tf.nn.space_to_depth(warped_frame, 4)
                # concatenate the lr_frame with the depth warped frame
                model_input = tf.concat([lr_frame, depth], axis=-1)
                # Implementing the model
                pre_conv = tf.layers.conv2d(inputs=model_input, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="pre_conv", kernel_initializer=tf.contrib.layers.xavier_initializer())
                pre_relu = tf.nn.relu(pre_conv, name="pre_relu")
                pre = pre_relu
                for conv_layer in range(10):
                    conv1 = tf.layers.conv2d(inputs=pre, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="conv_1_" + str(conv_layer), kernel_initializer=tf.contrib.layers.xavier_initializer())
                    relu = tf.nn.relu(conv1, name="relu_" + str(conv_layer))
                    conv2 = tf.layers.conv2d(inputs=relu, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="conv_2_" + str(conv_layer), kernel_initializer=tf.contrib.layers.xavier_initializer())
                    pre = conv2

                transpose_1 = tf.layers.conv2d_transpose(pre, 64, [3, 3], strides=2, padding='same', name="transpose_1", kernel_initializer=tf.contrib.layers.xavier_initializer())
                relu_1 = tf.nn.relu(transpose_1, name="relu_1")
                transpose_2 = tf.layers.conv2d_transpose(relu_1, 64, [3, 3], strides=2, padding='same', name="transpose_2", kernel_initializer=tf.contrib.layers.xavier_initializer())
                relu_2 = tf.nn.relu(transpose_2, name="relu_2")
                estimate = tf.layers.conv2d(inputs=relu_2, filters=3, kernel_size=[3, 3], padding="same", strides=1, name="estimate", kernel_initializer=tf.contrib.layers.xavier_initializer())
                return estimate

#        reshaped_lr = tf.reshape(lr_frame_input, [batch_size, frames_len, 1, height, width, channels])
        reshaped_lr = tf.transpose(lr_frame_input, [1, 0, 2, 3, 4])
        reshaped_lr = tf.reshape(reshaped_lr, [frames_len, batch_size, height, width, channels])


#        reshape_flow = tf.reshape(flow_input, [batch_size, frames_len, 1, height, width, flow_depth])
        reshaped_flow = tf.transpose(flow_input, [1, 0, 2, 3, 4])
        reshaped_flow = tf.reshape(reshaped_flow, [frames_len, batch_size, height, width, flow_depth])

        batch_input = tf.concat([reshaped_lr, reshaped_flow], axis=-1)

        # frames_len * batch_size * hr_h * hr_w * channels
        states = tf.scan(step, batch_input, initializer=tf.convert_to_tensor(np.zeros((batch_size, 256, 256, channels))))

        # batch_size * frames_len * hr_h * hr_w * channels
        output = tf.transpose(states, [1, 0, 2, 3, 4])

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, hr_frame_input), [1, 2, 3, 4]))
        self.train_op = tf.train.AdagradOptimizer(learning_rate=1e-4).minimize(self.loss)
        self.predict = output
        self.check_point_path = check_point_path

        self.lr_frame_input = lr_frame_input
        self.hr_frame_input = hr_frame_input
        self.flow_input = flow_input

        self.check_point_path = check_point_path

    ####
    # training
    def train(self, data_set, epochs=2000):
        # training session
        global_step = -1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            check_point = tf.train.get_checkpoint_state(self.check_point_path)
            saver = tf.train.Saver(max_to_keep=2)
            if check_point and check_point.model_checkpoint_path:
                logger.debug('Restored to a checkpoint stored at {}'.format(check_point.model_checkpoint_path))
                saver.restore(sess, check_point.model_checkpoint_path)
                global_step = int(check_point.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                logger.debug('No checkpoint is found for FRVSR to load')
            train_loss = 0
            for i in range(epochs):
                steps = 100
                progress_bar = trange(steps, desc='Training', leave=True)
                global_step = global_step + 1
                for j in progress_bar:
                    lr_frame_input, hr_frame_input, flow_input = data_set.next_data()
                    trial_max_count = 40
                    for trial in range(trial_max_count):
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict={
                            self.lr_frame_input:lr_frame_input,
                            self.hr_frame_input:hr_frame_input,
                            self.flow_input:flow_input
                        })
                    train_loss += train_loss_
                    progress_bar.set_description('last loss : {:.2f}, average loss: {:.2f}'.format(train_loss_, train_loss/(j + 1)))
                    progress_bar.refresh()
                logger.debug('[epoch:{:.0f}] Finished the current Epoch with average loss : {:.2f}'.format(i, train_loss/steps))
                saver.save(sess, self.check_point_path + 'frvsr.ckpt', global_step=global_step)
                train_loss = 0


    def test_inference(self, lr_inputs, flow_inputs, hr_inputs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            check_point = tf.train.get_checkpoint_state(self.check_point_path)
            saver = tf.train.Saver(max_to_keep=2)
            if check_point and check_point.model_checkpoint_path:
                logger.debug('Restored to a checkpoint stored at {}'.format(check_point.model_checkpoint_path))
                saver.restore(sess, check_point.model_checkpoint_path)
            else:
                logger.debug('No checkpoint is found for FRVSR to load')
            predict_hr, train_loss = sess.run([self.predict, self.loss], feed_dict={
                self.lr_frame_input:lr_inputs,
                self.hr_frame_input:hr_inputs,
                self.flow_input:flow_inputs
            })
        print("Training loss is {:.2}".format(train_loss))
        self.plot_inference(lr_inputs[0], predict_hr[0], hr_inputs[0])


    def plot_inference(self, lr, predict_hr, hr):

        for i in range(len(lr)):
            print(i)
            hr[i] = 255 * hr[i]
            hr[i] = hr[i].astype(np.uint8)

            lr[i] = 255 * lr[i]
            lr[i] = lr[i].astype(np.uint8)

            predict_hr[i] = 255 * np.clip(predict_hr[i], 0, 1)
            predict_hr[i] = predict_hr[i].astype(np.uint8)

            misc.imshow(hr[i])
            misc.imshow(predict_hr[i])
            misc.imshow(lr[i])
        print("After")
        print(hr[0])
        print(lr[0])
        print(predict_hr[0])

    def show_image(self, img):
        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        misc.imshow(img)
