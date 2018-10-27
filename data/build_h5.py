import cv2
from data.hdf5 import HDF5DatasetWriter
import os
import os.path as osp
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import logging
import sys
import config
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('build_h5')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

dataset_dir = config.DATASET_DIR
train_image_dir = osp.join(dataset_dir, config.TRAIN_IMAGE_DIR_NAME)
val_image_dir = osp.join(dataset_dir, config.VAL_IMAGE_DIR_NAME)
train_label_dir = osp.join(dataset_dir, config.TRAIN_LABEL_DIR_NAME)
val_label_dir = osp.join(dataset_dir, config.VAL_LABEL_DIR_NAME)
# train_data_writer = HDF5DatasetWriter(image_dims=(7900, 1024, 1024, 3),
#                                       label_dims=(7900, 256, 256, 7),
#                                       output_path='train_7900.hdf5')
val_data_writer = HDF5DatasetWriter(image_dims=(1976, 1024, 1024, 3),
                                    label_dims=(1976, 256, 256, 7),
                                    output_path='val_1976.hdf5')


def build(writers, image_dirs, label_dirs):
    for writer, image_dir, label_dir in zip(writers, image_dirs, label_dirs):
        image_filenames = os.listdir(image_dir)
        num_images = len(image_filenames)
        for image_filename, _ in zip(image_filenames, tqdm(range(num_images))):
            image_filepath = osp.join(image_dir, image_filename)
            logger.debug('Handling {} starts'.format(image_filepath))
            image = cv2.imread(image_filepath)
            image = preprocess_input(image, mode='tf')
            label_filename = image_filename[:-4] + '_gt.npy'
            label_filepath = osp.join(label_dir, label_filename)
            label = np.load(label_filepath)
            writer.add([image], [label])
            logger.debug('Handling {} ends'.format(image_filepath))
        writer.close()


writers = [val_data_writer]
image_dirs = [val_image_dir]
label_dirs = [val_label_dir]
build(writers, image_dirs, label_dirs)
