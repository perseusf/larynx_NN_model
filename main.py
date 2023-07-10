from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

augmentation_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

# initialize the model
model = unet()
model_checkpoint = ModelCheckpoint('unet_larynx.hdf5', monitor='loss', verbose=1, save_best_only=True)

# prepare videos
unpack_video('data/larynx/train', video_folder='video', image_folder='image', target_size=(256, 256))
unpack_tif('data/larynx/train', tif_folder='tifs', label_folder='label', target_size=(256, 256))

# train the model
training_dataset = training_dataset_generator(batch_size=2,
                                              train_path='data/larynx/train',
                                              image_folder='image',
                                              mask_folder='label',
                                              aug_dict=augmentation_args,
                                              save_to_dir=None)
model.fit_generator(training_dataset, steps_per_epoch=800, epochs=30, callbacks=[model_checkpoint])

# testing the model
test_dataset = test_dataset_generator("data/larynx/test")
results = model.predict_generator(test_dataset, 6, verbose=1)
saveResult("data/larynx/test", results)
