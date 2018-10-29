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

RESIZED_IMAGE_DIR = '/home/adam/.keras/datasets/text_detection/ICPR/resized_images'
TRAIN_IMAGE_DIR = '/home/adam/.keras/datasets/text_detection/ICPR/train_images'
VAL_IMAGE_DIR = '/home/adam/.keras/datasets/text_detection/ICPR/val_images'


def copy(num_files, src_dir, target_dir):
    logger.info('Before copying: num_images={}'.format(len(os.listdir(target_dir))))
    image_filenames = os.listdir(src_dir)
    # random.shuffle 会对原数组进行修改
    random.shuffle(image_filenames)
    to_copy_files = image_filenames[:num_files]
    for file in to_copy_files:
        shutil.copy(osp.join(src_dir, file), osp.join(target_dir, file))
    logger.info('After copying: num_images={}'.format(len(os.listdir(target_dir))))


copy(7900, RESIZED_IMAGE_DIR, TRAIN_IMAGE_DIR)
copy(1976, RESIZED_IMAGE_DIR, VAL_IMAGE_DIR)
