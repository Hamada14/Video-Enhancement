import tensorflow as tf

HR_INDEX = 2
LR_INDEX = 0
FLOW_INDEX = 1

class FRVSR():

    def __init__(self, batch_size, frames_len, height, width, channels, flow_depth, check_point_path):

        lr_frame_input = tf.placeholder(dtype=tf.int32)
        flow_input = tf.placeholder(dtype=tf.float32)
        hr_frame_input = tf.placeholder(dtype=tf.int32)


        def step(previous_output, current_input):

            lr_frame = current_input[LR_INDEX]
            flow = current_input[FLOW_INDEX]

            with tf.variable_scope("frvsr_model", reuse=tf.AUTO_REUSE):

                upscaled_flow = tf.image.resize_bilinear(flow, previous_output.shape)

                # warpping the last estimate with the flow
                warped_frame = tf.contrib.image.dense_image_warp(previous_output, upscaled_flow)

                # space to depth for warpped frame
                depth = tf.space_to_depth(warped_frame, 4)

                # concatenate the lr_frame with the depth warped frame
                model_input = tf.concat([lr_frame, depth])

                # Implementing the model
                pre_conv = tf.layers.conv2d(inputs=model_input, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="pre_conv")
                pre_relu = tf.nn.relu(pre_conv, name="pre_relu")
                pre = pre_relu
                for conv_layer in range(10):
                    conv1 = tf.layers.conv2d(inputs=pre, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="conv_1_" + str(conv_layer))
                    relu = tf.nn.relu(conv, name="relu_" + str(conv_layer))
                    conv2 = tf.layers.conv2d(inputs=relu, filters=64, kernel_size=[3, 3], padding="same", strides=1, name="conv_2_" + str(conv_layer))
                    pre = conv2

                transpose_1 = tf.nn.conv2d_transpose(pre, [3, 3], strides=2, padding='same', name="transpose_1")
                relu_1 = tf.nn.relu(transpose_1, name="relu_1")
                transpose_2 = tf.nn.conv2d_transpose(relu_1, [3, 3], strides=2, padding='same', name="transpose_2")
                relu_2 = tf.nn.relu(transpose_2, name="relu_2")

                estimate = tf.layers.conv2d(inputs=relu, filters=3, kernel_size=[3, 3], padding="same", strides=1, name="estimate")

                return estimate

            reshaped_lr = tf.reshape(lr_frame_input, [batch_size, frames_len, 1, height, width, channels])
            reshape_flow = tf.reshape(flow_input, [batch_size, frames_len, 1, height, width, flow_depth])
            batch_input = tf.concat([reshaped_lr, reshape_flow], 2)

            # frames_len * batch_size * hr_h * hr_w * channels
            states = tf.scan(step, tf.transpose(batch_input, [1, 0, 2, 3, 4]), initializer=np.zeros((batch_size, height, width, channels)))

            # batch_size * frames_len * hr_h * hr_w * channels
            output = tf.transpose(states, [1, 0, 2, 3, 4])

            losses = tf.map_fn(tf.nn.l2_loss, tf.to_float(hr_frame_input - output))
            self.loss = tf.reduce_mean(losses)
            self.train_op = tf.train.AdagradOptimizer(learning_rate=1e-4).minimize(loss)
            self.predict = predict
            self.check_point_path = check_point_path

    ####
    # training
    def train(self, epochs=100):
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(100):
                        lr_frame_input, hr_frame_input, flow_input = train_set.__next__()
                        batch_size = xs.shape[0]
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.xs_ : xs,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([2, batch_size, self.state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i,train_loss/100))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
            #
            # training ends here;
            #  save checkpoint
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
