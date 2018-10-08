import tensorflow as tf
import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt

import display

def build_model_input():
    
    print('Import dataset')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape, test_images.shape)

    print('Rescale images')
    train_images = train_images / 255.0
    test_images  = test_images / 255.0

    train_images = np.expand_dims(train_images, axis=-1)
    test_images  = np.expand_dims(test_images, axis=-1)

    # display.labeled_images(train_images, train_labels, CLASS_NAMES)

    return train_images, train_labels, test_images, test_labels

def build_model():
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3),  input_shape=(28,28,1), activation=tf.nn.relu),
        keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_images, train_labels, n_epochs=1):
    model.fit(train_images, train_labels, epochs=n_epochs)

def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

def display_predictions(models, x_test, y_test):

    prob_array = model.predict(x_test)
    display.image_prob_array(prob_array, y_test, x_test, CLASS_NAMES)
    plt.savefig('predictions.png')

if __name__ == '__main__':

    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Extract and preprocess data 
    x_train, y_train, x_test, y_test = build_model_input()
    # Build model
    model = build_model()
    # Train model
    train_model(model, x_train, y_train, n_epochs=5)
    # Evaluate accuracy
    evaluate_model(model, x_test, y_test)
    # Display predictions
    display_predictions(model, x_test, y_test)

    
