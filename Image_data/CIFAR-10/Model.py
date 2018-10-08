import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model(object):
    """docstring for Model"""
    def __init__(self, num_class=10):
        self.num_class  = num_class

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.sess = session

    def build_model(self, input_shape, y_out):
        
        tf.reset_default_graph()
        # Set input
        self._set_input(input_shape)
        # Set ouput (and graph)
        self._set_output(y_out)
        # Define losses and optimizers
        self._add_training_vars()

    def _set_input(self, input_shape):
        """ 
        input: 
        - input_shape: shape of the input tensor
        """
        # Setup input
        width, height, channel = input_shape
        self.X = tf.placeholder(tf.float32, [None,  width, height, channel])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

    def _set_output(self, y_out):
        self.y_out = y_out

    def _add_training_vars(self):

        mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.one_hot(self.y, self.num_class), logits=self.y_out))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        # Necessary if we want to use batch_normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(mean_loss)

        self.mean_loss = mean_loss
        self.train_step = train_step


    def run_model(self, x_data, y_data, is_training=False, 
                  num_epochs=1, batch_size=64, print_every=100):

        # Compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.y_out, axis=1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indices
        train_indices = np.arange(x_data.shape[0])
        np.random.shuffle(train_indices)

        # setting up variables we want to compute
        variables = [self.mean_loss, correct_prediction]
        if is_training:
            variables.append(self.train_step)
        else:
            variables.append(accuracy)

        iter_cnt = 0
        num_batches = int(np.ceil(x_data.shape[0] / batch_size))
        for e in range(num_epochs):

            correct = 0
            losses = []

            for i in range(num_batches):
                start_idx = (i * batch_size) 
                idx = train_indices[start_idx : start_idx + batch_size]

                feed_dict = {self.X: x_data[idx, :], 
                             self.y: y_data[idx],
                             self.is_training: is_training}

                actual_batch_size = y_data[idx].shape[0]

                loss, corr, _ = self.sess.run(variables, feed_dict=feed_dict)

                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)

                if is_training and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1: .3g} and accuracy of {2:.2g}"\
                            .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                iter_cnt += 1

            total_accuracy = correct / x_data.shape[0]
            total_loss = np.sum(losses) / x_data.shape[0]

            print("Epoch {0}, Overall loss = {1:.3g} and accuracy of {2:.3g}".format(e + 1, total_loss, total_accuracy))

        return total_accuracy, total_loss

