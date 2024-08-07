import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from keras.models import *
import keras.backend as K
from keras import layers
from keras import losses
from keras import callbacks
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
from data import test_dataset_generator
# from keras import backend as keras


IMG_SIZE = 256
seed=17

class PredictCallback(callbacks.Callback):
    def __init__(self, path):
        self.image_path = path
        
    def on_epoch_end(self, epoch, logs={}):
        image = test_dataset_generator('data\larynx\image_callback')
        y_pred = self.model.predict(image, 1, verbose=0)
        plt.imshow(y_pred[0], cmap='gray')
        plt.title(f'Prediction Visualization - Epoch: {epoch}')
        plt.savefig('model_train_images/test_epoch'+str(epoch))
        plt.close() 

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(
        factor=0.05,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=seed
  ),
  layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
        fill_mode='reflect',
        interpolation='bilinear',
        seed=seed
  ),
  layers.RandomZoom(
        height_factor=0.01,
        width_factor=0.01,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=seed
  ),
  layers.RandomContrast(
        factor=0.05,
        seed=seed)
])


def dice_coefficient(y_true, y_pred, smooth=0.001):       
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    return ((2. * intersection) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth))


def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)



def unet(pretrained_weights=None, input_size=(IMG_SIZE, IMG_SIZE, 1)):

    """
    U-Net model for image segmentation using convolutional neural networks.

    Args:
        pretrained_weights (str): path to the weights file to use for initializing the model.
                                  If None, the model is initialized with random weights.
        input_size (tuple): size of the input image as a tuple (height, width, channels).

    Returns:
        A Keras model object representing the U-Net model.

    The U-Net architecture consists of an encoder part that gradually reduces the spatial
    dimensions of the input image, and a decoder part that gradually increases them back to the
    original size. Skip connections are used to combine the output of the encoder with the
    corresponding input of the decoder, allowing the model to recover fine-grained details. The
    final layer uses a sigmoid activation function to predict a binary segmentation mask for
    each pixel of the input image.

    The model is compiled with the binary crossentropy loss function and the Adam optimizer with a
    learning rate of 1e-4. The accuracy metric is also computed during training.
    """


    inputs = tf.keras.Input(input_size)

    # augmented_inputs = data_augmentation(inputs)
    augmented_inputs = inputs

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(augmented_inputs)
    # conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # conv4 = layers.BatchNormalization()(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # conv5 = layers.BatchNormalization()(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # conv6 = layers.BatchNormalization()(conv6)


    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # conv8 = layers.BatchNormalization()(conv8)


    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = layers.BatchNormalization()(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=dice_loss, metrics=[dice_coefficient])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
