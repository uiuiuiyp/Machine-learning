import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau
from keras import models, layers
from sklearn.metrics import confusion_matrix

import util

TRAIN_DIR = 'training/'
TEST_DIR  = 'validation/'
BATCH_SIZE = 32

def build_model_input():
    
    print("Import class names")
    cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
    info = pd.read_csv("monkey_labels.txt", names=cols, skiprows=1)    
    global LABELS
    LABELS = info['Common Name']

    util.display_image('training/n0/n0018.jpg')

    print("Data augmentation/Preprocessing")
    height, width, channels = 299, 299, 3

    train_datagen   = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                        target_size=(height, width),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')
    
    test_datagen   = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(TEST_DIR, 
                                                      target_size=(height, width),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='categorical')

    print("Import pretrained Inception module")
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(height, width, channels))

    print("Extract features")
    train_features, train_labels = util.extract_features(1097, BATCH_SIZE, train_generator, base_model)
    test_features, test_labels   = util.extract_features(272, BATCH_SIZE, test_generator, base_model)

    return train_features, train_labels, test_features, test_labels

def build_model():

    # Define callback
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                             factor=0.1,
                                             patience=2,
                                             cooldown=2,
                                             min_lr=1e-6,
                                             verbose=1)
    callbacks = [reduce_learning_rate]

    model = models.Sequential([
        layers.AveragePooling2D(input_shape=(10,10,2048)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, callbacks

def train_model(model, x_train, y_train, n_epochs=1):
    
    print("Training model")
    history = model.fit(x_train, y_train, verbose=0, epochs=n_epochs,
                        batch_size=BATCH_SIZE, shuffle=True,
                        validation_split=0.1, callbacks=callbacks)

    return history

def display_accuracies(history):

    # Unpack history
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    n_epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.title('Training and validation accuracy')
    plt.plot(n_epochs, acc, 'red', label='Training acc')
    plt.plot(n_epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    plt.savefig('accuracies.png')

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(n_epochs, loss, 'red', label='Training loss')
    plt.plot(n_epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    plt.savefig('losses.png')

def display_confusion_matrix(model, x_test, y_test):
    prob_array = model.predict(x_test)

    y_pred = [i.argmax() for i in prob_array]
    y_true = [i.argmax() for i in y_test]

    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    util.plot_confusion_matrix(cm, normalize=True, target_names=LABELS)
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    
    # Apply Inception module to generate features to the images
    x_train, y_train, x_test, y_test =  build_model_input()
    # Build model
    model, callbacks = build_model()
    # Fit the model
    history = train_model(model, x_train, y_train, n_epochs=30)
    # Display results
    display_accuracies(history)
    display_confusion_matrix(model, x_test, y_test)

    plt.show()
