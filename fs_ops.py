import shutil
import os
import random
import os.path as osp
import logging
import sys

logger = logging.getLogger('label')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

TRAIN_IMAGE_DIR = '/home/adam/.keras/datasets/text_detection/ICPR/train_images'
VAL_IMAGE_DIR = '/home/adam/.keras/datasets/text_detection/ICPR/val_images'


def copy_val_to_train(num_files):
    val_image_filenames = os.listdir(VAL_IMAGE_DIR)
    # random.shuffle 会对原数组进行修改
    random.shuffle(val_image_filenames)
    to_copy_files = val_image_filenames[:num_files]
    for file in to_copy_files:
        shutil.move(osp.join(VAL_IMAGE_DIR, file), osp.join(TRAIN_IMAGE_DIR, file))


logger.info('Before copying: num_val_images={}'.format(len(os.listdir(VAL_IMAGE_DIR))))
logger.info('Before copying: num_train_images={}'.format(len(os.listdir(TRAIN_IMAGE_DIR))))
copy_val_to_train(4928)
logger.info('After copying: num_val_images={}'.format(len(os.listdir(VAL_IMAGE_DIR))))
logger.info('After copying: num_train_images={}'.format(len(os.listdir(TRAIN_IMAGE_DIR))))
