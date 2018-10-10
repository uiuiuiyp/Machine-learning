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
    
    print("Build graph")
    model = Model(num_class=NUM_CLASS)
    model.set_input(input_shape)

    input_layer = model.X
    layer = tf.layers.conv2d(input_layer, 32, (3,3), strides=1, padding='same', activation=tf.nn.relu)
    layer = tf.layers.max_pooling2d(layer, 2, 2)
    layer = inception_module(layer)
    layer = inception_module(layer)
    layer = tf.layers.max_pooling2d(layer, 2, 2)
    layer = inception_module(layer)
    layer = inception_module(layer)
    layer = tf.layers.max_pooling2d(layer, 2, 2)
    layer = inception_module(layer)
    layer = inception_module(layer)
    layer = tf.layers.average_pooling2d(layer, 4, 4)

    layer_dim = layer.shape
    layer_flat = tf.reshape(layer, [-1, tf.reduce_prod(layer_dim[1:])])
    print(layer_flat.shape)
    layer_flat = tf.layers.dropout(layer_flat, rate=0.5, training=model.is_training)
    layer_flat = tf.layers.dense(layer_flat, layer_flat.shape[1]//2, activation=tf.nn.relu)

    y_out = tf.layers.dense(layer_flat, 10)

    # Define the tensorflow graph
    model.build_model(y_out)

    return model

def bottleneck(input_layer, channel):

    return tf.layers.conv2d(input_layer, channel, (1,1), strides=1, padding='same')

def inception_module(input_layer):

    N, width, height, channel = input_layer.shape

    layer_1x1  = bottleneck(input_layer, channel//3)
    reduce_3x3 = bottleneck(input_layer, channel//2)
    layer_3x3  = tf.layers.conv2d(reduce_3x3, channel//3*2, (3,3), strides=1, padding='same', activation=tf.nn.relu)
    reduce_5x5 = bottleneck(input_layer, channel//12)
    layer_5x5  = tf.layers.conv2d(reduce_5x5, channel//5, (5,5), strides=1, padding='same', activation=tf.nn.relu)
    pooling    = tf.layers.max_pooling2d(input_layer, (3,3), strides=1, padding='same')
    pool_proj  = bottleneck(pooling, channel//5)

    output_layer = tf.concat([layer_1x1, layer_3x3, layer_5x5, pool_proj], axis=3)

    print(layer_1x1.shape, layer_3x3.shape, layer_5x5.shape, pool_proj.shape)
    # print(output_layer.shape)

    return output_layer

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
