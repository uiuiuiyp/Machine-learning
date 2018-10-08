import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Model import Model
import data_utils

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs to train the model")
    args = vars(parser.parse_args())
    n_epochs = args.pop('n_epochs')

    print('num_epochs', n_epochs)

    return n_epochs

def build_model_input(num_training=49000, num_validation=1000, num_test=10000):
    
    print('Load data')
    cifar10_dir = 'datasets'
    x_train, y_train, x_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

    print('Subsample the data')
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test

def build_model(input_shape):
    
    print("Initialize model")
    model = Model(num_class=NUM_CLASS)
    model.build_model(input_shape)

    print("Building graph")
    input_layer = model.X

    layer = tf.layers.conv2d(input_layer, 32, (3,3), strides=1, padding='valid', activation=tf.nn.relu)
    layer = tf.layers.conv2d(layer, 32, (3,3), strides=1, padding='valid', activation=tf.nn.relu)
    layer = tf.layers.max_pooling2d(layer, 2, 2)

    layer = tf.layers.conv2d(layer, 64, (3,3), strides=1, padding='valid', activation=tf.nn.relu)
    layer = tf.layers.conv2d(layer, 64, (3,3), strides=1, padding='valid', activation=tf.nn.relu)
    layer = tf.layers.max_pooling2d(layer, 2, 2)
    
    layer_dim = layer.shape
    layer_flat = tf.reshape(layer, [-1, tf.reduce_prod(layer_dim[1:])])
    layer_flat = tf.layers.dense(layer_flat, 100, activation=tf.nn.relu)
    layer_flat = tf.layers.dropout(layer_flat, rate=0.5, training=self.is_training)

    y_out = tf.layers.dense(layer_flat, 10)

    # Set the output and ultimately determine the tensorflow graph
    model.set_output(y_out)

    return model

def train_model(model, x_train, y_train, num_epochs=1):
    
    model.get_session()
    model.sess.run(tf.global_variables_initializer())

    print("Training the model")
    loss, accuracy = model.run_model(x_train, y_train, is_training=True, num_epochs=num_epochs)

    return loss, accuracy

def evaluate_model(model, x_test, y_test):
    
    print("Evaluate the model with validation set")
    loss, accuracy = model.run_model(x_test, y_test)

if __name__ == '__main__':

    num_epochs = parse_arguments()

    NUM_CLASS = 10

    # Extract and preprocess data 
    x_train, y_train, x_val, y_val, x_test, y_test = build_model_input()

    # Build model
    input_shape = x_train.shape[1:]
    model = build_model(input_shape)
    # Train model
    train_model(model, x_train, y_train, num_epochs=num_epochs)
    # Evaluate accuracy
    evaluate_model(model, x_test, y_test)
