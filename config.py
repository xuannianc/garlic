import os

train_task_id = '3T736'
initial_epoch = 0
epoch_num = 24
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
load_weights = False
# loss 的权重
LAMBDA_INSIDE_SHRINK_QUAD_SCORE_LOSS = 4.0
LAMBDA_INSIDE_END_QUADS_SCORE_LOSS = 1.0
LAMBDA_VERTICES_COORD_LOSS = 1.0

total_img = 10000
VAL_SPLIT_RATIO = 0.3
BATCH_SIZE = 20
DATASET_DIR = '/home/adam/.keras/datasets/text_detection/ICPR'
ORIGIN_IMAGE_DIR_NAME = 'image_10000'
ORIGIN_LABEL_DIR_NAME = 'txt_10000'
RESIZED_IMAGE_DIR_NAME = 'resized_images'
# .npy 的文件
RESIZED_LABEL_DIR_NAME = 'resized_labels'
# train_image 是 resized_images 是 0.7
TRAIN_IMAGE_DIR_NAME = 'train_images'
TRAIN_LABEL_DIR_NAME = 'train_labels'
VAL_IMAGE_DIR_NAME = 'val_images'
VAL_LABEL_DIR_NAME = 'val_labels'
DRAW_GT_QUAD_IMAGE_DIR_NAME = 'draw_gt_quad_images'
DRAW_ACT_IMAGE_DIR_NAME = 'draw_act_images'
SAVE_RESIZED_IMAGE = False
SAVE_RESIZED_LABEL = True
SAVE_TRAIN_VAL_IMAGE = False
SAVE_DRAW_GT_QUAD_IMAGE = False
DRAW_GT_QUAD = False
DRAW_ACT = True
# in paper it's 0.3, maybe too large to this problem
SHRINK_RATIO = 0.2
# pixels between 0.2 and 0.6 are side pixels
SHRINK_SIDE_RATIO = 0.6
EPSILON = 1e-7
FEATURE_LAYERS_RANGE = [4, 3, 2, 1]
# vgg block2_pool 的输出就是 f4, 每个像素相当于原来的 4 个像素
PIXEL_SIZE = 2 ** 2
LOCK_LAYERS = False

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5' \
                                % train_task_id

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True

IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
