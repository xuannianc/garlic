import os

train_task_id = '3T736'
initial_epoch = 0
epoch_num = 24
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
load_weights = False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
VALIDATION_SPLIT_RATIO = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - VALIDATION_SPLIT_RATIO) // batch_size
validation_steps = total_img * VALIDATION_SPLIT_RATIO // batch_size

DATASET_DIR = '/home/adam/.keras/datasets/text_detection/ICPR'
ORIGIN_IMAGE_DIR_NAME = 'image_10000'
ORIGIN_LABEL_DIR_NAME = 'txt_10000'
CURRENT_IMAGE_DIR_NAME = 'images_%s' % train_task_id
CURRENT_LABEL_DIR_NAME = 'labels_%s' % train_task_id
DRAW_GT_QUAD_IMAGE_DIR_NAME = 'draw_gt_quad_images_%s' % train_task_id
DRAW_ACT_IMAGE_DIR_NAME = 'draw_act_images_%s' % train_task_id
SAVE_RESIZED_IMAGE = True
DRAW_GT_QUAD = True
draw_act_quad = True
VAL_FILENAME = 'val_%s.txt' % train_task_id
TRAIN_FILENAME = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe too large to this problem
SHRINK_RATIO = 0.2
# pixels between 0.2 and 0.6 are side pixels
SHRINK_SIDE_RATIO = 0.6
EPSILON = 1e-7

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
PIXEL_SIZE = 2 ** feature_layers_range[-1]
locked_layers = False

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                % train_task_id

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True

IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
