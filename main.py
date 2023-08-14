import tensorflow as tf
import datetime, os, shutil
from model import *
from model import PredictCallback
from data import *
import splitfolders


print(tf.config.list_physical_devices('GPU'))

model = unet()
# model_checkpoint = ModelCheckpoint('unet_larynx_{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=True)
model_checkpoint = ModelCheckpoint('unet_larynx.hdf5', monitor='val_loss', verbose=1, save_best_only=True)


training_dataset = training_dataset_generator(2,
                                              'data/larynx/train/train_dataset/',
                                              'image',
                                              'label',
                                              save_to_dir=None)

val_dataset = training_dataset_generator(2,
                                              'data/larynx/train/validation_dataset/',
                                              'image',
                                              'label',
                                              save_to_dir=None)

# Set up TensorBoard callback
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

path_to_image = 'data\larynx\image_callback'
epoch_predict = PredictCallback(path_to_image)

# Train the model using the training dataset and TensorBoard callback
history = model.fit(training_dataset, 
          steps_per_epoch=100, 
          epochs=5000, 
          callbacks=[model_checkpoint, tensorboard_callback, early_stopping, epoch_predict],
          validation_data=val_dataset,
          validation_steps=100
        )




















