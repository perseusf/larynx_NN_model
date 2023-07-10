from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import multipagetiff as mtif
from matplotlib import pyplot as plt


Larynx = [128, 128, 128]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Larynx, Unlabelled])


def unpack_video(train_path, video_folder, image_folder, target_size=(256, 256)):
    """
    Unpack videos (.avi) and saves the frames to folder "image"

    Args:
        video_folder (str): path to the folder containing the videos (.avi)
        image_folder (str): path to the folder where the frames will be saved

    Returns:
        None
    """

    # Check if the frame_folder directory exists, create one if it doesn't exist
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    # Get all the video file names from the video_folder directory
    video_files = [f for f in os.listdir(train_path + '/' + video_folder)]

    # Loop through the video files
    for video_file in video_files:
        # Open the video file
        cap = cv2.VideoCapture(os.path.join(train_path, video_folder, video_file))

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through all the frames in the video
        for i in range(total_frames):
            # Read the frame from the video
            ret, frame = cap.read()

            # Check if the frame was successfully read
            if ret:
                # Check the shape of the frame
                if frame.shape != target_size:
                    frame = cv2.resize(frame, target_size)

                # Make frame grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Save the frame to the frame_folder directory
                frame_name = os.path.splitext(video_file)[0] + '_' + str(i) + '.png'
                cv2.imwrite(os.path.join(train_path, image_folder, frame_name), frame)

            else:
                break

        # Release the video object
        cap.release()



def unpack_tif(train_path, tif_folder, label_folder, target_size=(256, 256)):
    """
    Reads tif files from a folder, creates frames from them,
    and saves the frames to another folder.

    Args:
        tif_folder (str): Path to folder containing tif files
        label_folder (str): Path to folder where frames will be saved

    Returns:
        None
    """

    # Check if the frame_folder directory exists, create one if it doesn't exist
    if not os.path.isdir(os.path.join(train_path, label_folder)):
        os.mkdir(os.path.join(train_path, label_folder))

    # Get list of all .tif files in folder
    tif_files = [f for f in os.listdir(train_path + '/' + tif_folder)]
    # Loop through each .tif file
    for tif_file in tif_files:

        # Open .tif file and convert to grayscale
        img = mtif.read_stack(os.path.join(train_path, tif_folder, tif_file))
        img_gray = np.array(img)


        for i in range(0, img_gray.shape[0]):
            # Extract frame and save to disk
            frame = np.array(img_gray[i, :, :])
            if frame.shape != target_size:
                frame = cv2.resize(frame, target_size)
            frame_name = f"{os.path.splitext(os.path.basename(tif_file))[0]}_{i}.png"
            save_path = os.path.join(train_path, label_folder, frame_name)
            plt.imsave(save_path, frame, cmap='binary_r')
            # cv2.imwrite(save_path, frame)


def adjust_data(img, mask, flag_multi_class, num_class):

    """
    Adjusts the given image and mask according to the specified parameters.

    Args:
        img (numpy.ndarray): The input image.
        mask (numpy.ndarray): The input mask.
        flag_multi_class (bool): A flag indicating whether the mask contains multiple classes or not.
        num_class (int): The number of classes in the mask.

    Returns:
        A tuple of the adjusted image and mask.
    """

    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def training_dataset_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                               mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                               flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    """
    Generates batches of augmented data for training a model.

    Args:
        batch_size (int): The batch size.
        train_path (str): The path to the training data.
        image_folder (str): The name of the folder containing the input images.
        mask_folder (str): The name of the folder containing the input masks.
        aug_dict (dict): A dictionary of augmentation parameters for the images and masks.
        image_color_mode (str): The color mode of the input images. Default is "grayscale".
        mask_color_mode (str): The color mode of the input masks. Default is "grayscale".
        image_save_prefix (str): The prefix to use for saving augmented images. Default is "image".
        mask_save_prefix (str): The prefix to use for saving augmented masks. Default is "mask".
        flag_multi_class (bool): A flag indicating whether the masks contain multiple classes or not. Default is False.
        num_class (int): The number of classes in the masks. Default is 2.
        save_to_dir (str): The directory to save the augmented images and masks to. Default is None.
        target_size (tuple): The target size of the images and masks. Default is (256, 256).
        seed (int): The random seed. Default is 1.

    Returns:
        A generator that yields batches of augmented data.
    """

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield img, mask


def test_dataset_generator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):

    """
    Generates test images as input for the model.

    Args:
        test_path (str): Path to the directory containing the test images.
        num_image (int): The number of images to generate.
        target_size (tuple): A tuple representing the size to which the images should be resized.
        flag_multi_class (bool): A boolean indicating whether the images have multiple classes.
        as_gray (bool): A boolean indicating whether the images should be converted to grayscale or not.

    Yields:
        ndarray: A numpy array containing the test images.
    """

    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):

    """
    Generate NumPy arrays from a set of training images and their corresponding masks.

    Args:
        image_path (str): Path to the directory containing the input images.
        mask_path (str): Path to the directory containing the target masks.
        flag_multi_class (bool): Whether the masks have multiple classes or not.
        num_class (int): Number of classes in the target masks, if `flag_multi_class` is True.
        image_prefix (str): Prefix to use when looking for input image files.
        mask_prefix (str): Prefix to use when looking for target mask files.
        image_as_gray (bool): Whether to load input images as grayscale or not.
        mask_as_gray (bool): Whether to load target masks as grayscale or not.

    Returns:
        Two NumPy arrays, one containing the preprocessed input images and one containing the preprocessed target masks.

    Raises:
        None.
    """

    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    """
    Visualize label masks with color-coded classes.

    Args:
        num_class (int): The number of classes in the label mask.
        color_dict (dict): A dictionary mapping class indices to RGB color values.
        img (ndarray): A 2D or 3D numpy array representing the label mask.

    Returns:
        ndarray: A 3D numpy array representing the color-coded label mask.

    Note:
        This function assumes that the label mask contains class indices ranging from 0 to num_class - 1,
        and the color_dict contains RGB color values normalized to the range [0, 255].
    """

    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    """
    Save the predicted masks as images to the specified directory.

    Args:
        - save_path (str): The path to the directory where the predicted masks should be saved.
        - npyfile (numpy.ndarray): The predicted masks to save as images.
        - flag_multi_class (bool): A flag indicating whether the problem is multi-class or binary.
        - num_class (int): The number of classes in the multi-class problem.

    Returns:
        - None
    """

    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
