import numpy as np
import os
import random
from tqdm import tqdm
import config
from label import shrink
import logging
import sys
import cv2
import os.path as osp
import shutil

logger = logging.getLogger('preprocess')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def batch_sort_xylist(xy_list_array):
    sorted_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        sorted_xy_list_array[i] = sort_xylist(xy_list)
    return sorted_xy_list_array


def sort_xylist(xy_list):
    """
    先找最小 x 的点, 作为返回值的第 0 个元素
    然后找最小 x 的点对角线的点, 作为返回值的第 2 个元素
    然后对角线下方的点, 作为返回值的第 1 个元素
    然后对角线上方的点, 作为返回值的第 3 个元素
    最后调整把左上方的点作为返回值的第　0 个元素, 其他按逆时针移动
    总结起来就是: 先找到最小 x 的点, 然后按逆时针找到其他三个点, 最后调整把左上方的点作为返回值的第　0 个元素, 其他按逆时针移动
    :param xy_list: shape 为 (4, 2) 的数组,ICPR 的数据集中依次存放左下方的点和按逆时针方向的其他三个点
    :return:
    """
    # sorted_xy_list 的 shape 为 (4, 2)
    sorted_xy_list = np.zeros_like(xy_list)
    first_vertex_idx, second_vertex_idx, third_vertex_idx, fourth_vertex_idx = None, None, None, None
    ############################### 找最小 x 的点,其下标复制给 first_vertex_idx ########################################
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    # 四个点的坐标按 x 进行排序, 如果 x 相等, 按 y 排序
    # np.argsort 返回结果的 shape 和原来一样, 每个元素的值表示该位置对应的原数组中的元素的序号
    # 参见 https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argsort.html
    sorted_idxes = np.argsort(xy_list, axis=0)
    # 有最小 x 的元素在原数组中的序号
    xmin1_idx = sorted_idxes[0, 0]
    # 有倒数第二小的 x 的元素在原数组的序号
    xmin2_idx = sorted_idxes[1, 0]
    # 如果最小的两个 x 相等, 比较 y
    if xy_list[xmin1_idx, 0] == xy_list[xmin2_idx, 0]:
        if xy_list[xmin1_idx, 1] <= xy_list[xmin2_idx, 1]:
            sorted_xy_list[0] = xy_list[xmin1_idx]
            first_vertex_idx = xmin1_idx
        else:
            sorted_xy_list[0] = xy_list[xmin2_idx]
            first_vertex_idx = xmin2_idx
    else:
        # 把有最小 x 的元素放在  sorted_xy_list 的下标 0 的位置
        sorted_xy_list[0] = xy_list[xmin1_idx]
        first_vertex_idx = xmin1_idx
    ##################################### 找到第 0 个顶点的对角线的点　############################################
    # connect the first point to the third point on the other side of
    # the line with the middle slope
    other_vertex_idxes = list(range(4))
    other_vertex_idxes.remove(first_vertex_idx)
    # k 存放第一个点和其他点的斜率
    k = np.zeros((len(other_vertex_idxes),))
    for i, other_vertex_idx in enumerate(other_vertex_idxes):
        k[i] = (xy_list[other_vertex_idx, 1] - xy_list[first_vertex_idx, 1]) \
               / (xy_list[other_vertex_idx, 0] - xy_list[first_vertex_idx, 0] + config.EPSILON)
    # 防止在同一条直线
    if k[0] == k[1] and k[0] == k[2]:
        logger.warning('{} is invalid'.format(xy_list))
        return np.zeros_like(xy_list)
    # ｋ_mid_idx 是三个斜率的中间值的下标
    k_mid_idx = np.argsort(k)[1]
    # 中间值斜率
    k_mid = k[k_mid_idx]
    third_vertex_idx = other_vertex_idxes[k_mid_idx]
    sorted_xy_list[2] = xy_list[third_vertex_idx]
    ##################################### 找到其他两个点 ##################################################
    # determine the second point which on the bigger side of the middle line
    other_vertex_idxes.remove(third_vertex_idx)
    # 对角线的　y = k * x + b 的　b
    b_mid = xy_list[first_vertex_idx, 1] - k_mid * xy_list[first_vertex_idx, 0]
    for i, other_vertex_idx in enumerate(other_vertex_idxes):
        # delta_y = y - (k * x + b)
        # 根据　delta_y 判断该点是在对角线上方还是下方
        # 下方的点作为第 1 个点, 上方的点作为第 3 个点
        delta_y = xy_list[other_vertex_idx, 1] - (k_mid * xy_list[other_vertex_idx, 0] + b_mid)
        if delta_y > 0:
            second_vertex_idx = other_vertex_idx
        else:
            fourth_vertex_idx = other_vertex_idx
    if second_vertex_idx is None:
        logger.warning('Cannot find second_vertex_idx')
        return np.zeros_like(xy_list)
    if fourth_vertex_idx is None:
        logger.warning('Cannot find fourth_vertex_idx')
        return np.zeros_like(xy_list)
    sorted_xy_list[1] = xy_list[second_vertex_idx]
    sorted_xy_list[3] = xy_list[fourth_vertex_idx]
    ############################### 把左上方的点作为第一个点,按逆时针得到其他点　######################################
    # compare slope of 02 and 13, determine the final order
    k02 = k_mid
    k13 = (xy_list[second_vertex_idx, 1] - xy_list[fourth_vertex_idx, 1]) / (
            xy_list[second_vertex_idx, 0] - xy_list[fourth_vertex_idx, 0] + config.EPSILON)
    if k02 < k13:
        #  3_________2
        #  /        /
        # /________/
        # 0         1
        # 调整 3->0,2->3,1->2,0->1
        tmp_x, tmp_y = sorted_xy_list[3, 0], sorted_xy_list[3, 1]
        for i in range(2, -1, -1):
            sorted_xy_list[i + 1] = sorted_xy_list[i]
        sorted_xy_list[0, 0], sorted_xy_list[0, 1] = tmp_x, tmp_y
        # 调整后
        #  0_________3
        #  /        /
        # /________/
        # 1         2
    return sorted_xy_list


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    Args:
        min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
                   this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image
                    of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

    Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
                be inserted in the returned image. If so, this window is the
                coordinates of the image part of the full image (excluding
                the padding). The x2, y2 pixels are not included.
                window 的意思其实就是原 image 在 resized_image 中的位置
                UNCLEAR: 为什么不包含 x2,y2?
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (0, 0, h, w) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    # 三个维度的开始和结束处的 padding
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # 是否指定了 min_dim
    if min_dim:
        # Scale up but not down
        # 如果 h,w 的较小值大于 min_dim, scale=1.
        # 如果 h,w 的较小值小于 min_dim, scale=min_dim/min(h,w)>1
        scale = max(1, min_dim / min(h, w))
    # 是否指定了 min_scale
    if min_scale and scale < min_scale:
        scale = min_scale

    # 是否指定了 max_dim
    if max_dim and mode == "square":
        image_max_dim = max(h, w)
        # Does it exceed max dim?
        # 如果 h,w 的较大值乘以之前得到的 scale 超过了 max_dim, 重新定义 scale=max_dim/max(h,w)
        if round(image_max_dim * scale) > max_dim:
            scale = max_dim / image_max_dim

    # Resize image using bilinear interpolation
    # NOTE: round 四舍六入, 等于 5 时取就近的偶数, 默认不保留小数位,直接转成 int
    if scale != 1:
        # image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        # Get new height and width
        h, w = image.shape[:2]

    # Need padding or cropping?
    if mode == "square":
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # image 的 shape 经过 pad 后会变成 (max_dim, max_dim, 3)
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def every_image_has_corresponding_label(image_dir, label_dir):
    for image_filename in os.listdir(image_dir):
        label_filename = image_filename[:-4] + '.npy'
        label_filepath = osp.join(label_dir, label_filename)
        if not osp.exists(label_filepath):
            return False
    image_filenames = os.listdir(image_dir)
    num_images = len(image_filenames)
    logger.info('Found {} images.'.format(num_images))
    label_filenames = os.listdir(label_dir)
    num_labels = len(label_filenames)
    logger.info('Found {} labels.'.format(num_labels))
    return True


def preprocess():
    dataset_dir = config.DATASET_DIR
    # 存放原来的图片
    origin_image_dir = os.path.join(dataset_dir, config.ORIGIN_IMAGE_DIR_NAME)
    # 存放原来的 txt 标注
    origin_label_dir = os.path.join(dataset_dir, config.ORIGIN_LABEL_DIR_NAME)
    if config.SAVE_RESIZED_IMAGE:
        # 存放 resize 后的图片
        resized_image_dir = os.path.join(dataset_dir, config.RESIZED_IMAGE_DIR_NAME)
        if not os.path.exists(resized_image_dir):
            logger.info('resized_image_dir:{} does not exist, then creating'.format(resized_image_dir))
            os.mkdir(resized_image_dir)
        else:
            logger.info('resized_image_dir:{} already exists, then deleting and creating'.format(resized_image_dir))
            shutil.rmtree(resized_image_dir)
            os.mkdir(resized_image_dir)
    if config.SAVE_RESIZED_LABEL:
        # 存放 resize 后的标注
        resized_label_dir = os.path.join(dataset_dir, config.RESIZED_LABEL_DIR_NAME)
        if not os.path.exists(resized_label_dir):
            logger.info('resized_label_dir:{} does not exist, then creating'.format(resized_label_dir))
            os.mkdir(resized_label_dir)
        else:
            logger.info('resized_label_dir:{} already exists, then deleting and creating'.format(resized_label_dir))
            shutil.rmtree(resized_label_dir)
            os.mkdir(resized_label_dir)
    if config.SAVE_DRAW_GT_QUAD_IMAGE:
        # 存放画有 gt quad 的 resized_image
        draw_gt_quad_image_dir = os.path.join(dataset_dir, config.DRAW_GT_QUAD_IMAGE_DIR_NAME)
        if not os.path.exists(draw_gt_quad_image_dir):
            logger.info('draw_gt_quad_image_dir:{} does not exist, then creating'.format(draw_gt_quad_image_dir))
            os.mkdir(draw_gt_quad_image_dir)
        else:
            logger.info('draw_gt_quad_image_dir:{} already exists, then deleting and creating'.format(
                draw_gt_quad_image_dir))
            shutil.rmtree(draw_gt_quad_image_dir)
            os.mkdir(draw_gt_quad_image_dir)
    origin_image_filenames = os.listdir(origin_image_dir)
    num_origin_images = len(origin_image_filenames)
    logger.info('Found {} origin images.'.format(num_origin_images))
    for origin_image_filename, _ in zip(origin_image_filenames, tqdm(range(num_origin_images))):
        origin_image_filepath = os.path.join(origin_image_dir, origin_image_filename)
        logger.debug('Handling {} starts'.format(origin_image_filepath))
        origin_image = cv2.imread(origin_image_filepath)
        # cv2.namedWindow('origin_image', cv2.WINDOW_NORMAL)
        # cv2.imshow('origin_image', origin_image)
        # cv2.waitKey(0)
        if origin_image is None:
            logger.warning('Reading {} failed'.format(origin_image_filepath))
            continue
        image, window, scale, padding, crop = resize_image(origin_image,
                                                           min_dim=config.IMAGE_MIN_DIM,
                                                           max_dim=config.IMAGE_MAX_DIM,
                                                           mode=config.IMAGE_RESIZE_MODE)
        origin_label_filename = origin_image_filename[:-4] + '.txt'
        origin_label_filepath = os.path.join(origin_label_dir, origin_label_filename)
        if osp.exists(origin_label_filepath):
            with open(origin_label_filepath, 'r') as f:
                annotation_list = f.readlines()
        else:
            logger.warning('{} does not exist'.format(origin_label_filepath))
            continue
        # draw ground truth quad on image
        draw_gt_quad_image = image.copy()
        num_annotations = len(annotation_list)
        xy_list_array = np.zeros((num_annotations, 4, 2))
        for idx, annotation in enumerate(annotation_list):
            annotation_columns = annotation.strip().split(',')
            annotation_array = np.array(annotation_columns)
            xy_list = np.reshape(annotation_array[:8].astype(np.float64), (4, 2))
            xy_list = xy_list * scale
            # 所有 x 加上 left_padding
            xy_list[:, 0] += padding[1][0]
            # 所有 y 加上 top_padding
            xy_list[:, 1] += padding[0][0]
            # 希望得到从左上方的点开始逆时针旋转得到的所有点的坐标
            xy_list = sort_xylist(xy_list)
            xy_list_array[idx] = xy_list
            _, shrink_xy_list, _ = shrink(xy_list, config.SHRINK_RATIO)
            shrink_long_edges_xy_list, _, first_long_edge_idx = shrink(xy_list, config.SHRINK_SIDE_RATIO)
            if config.DRAW_GT_QUAD:
                cv2.polylines(draw_gt_quad_image, [xy_list.astype('int').reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.polylines(draw_gt_quad_image, [shrink_xy_list.astype('int').reshape(-1, 1, 2)], True, (255, 0, 0),
                              2)
                # 左右长上下短的情况
                # ----------------
                # |//////////////|
                # ----------------
                # |              |
                # |              |
                # |              |
                # ----------------
                # |//////////////|
                # ----------------
                if first_long_edge_idx == 0:
                    # 上端阴影
                    cv2.polylines(draw_gt_quad_image,
                                  [np.array([xy_list[0], shrink_long_edges_xy_list[0], shrink_long_edges_xy_list[3],
                                             xy_list[3]]).astype('int').reshape(-1, 1, 2)],
                                  True,
                                  (255, 255, 0), 2)
                    # 下端阴影
                    cv2.polylines(draw_gt_quad_image,
                                  [np.array([shrink_long_edges_xy_list[1], xy_list[1], xy_list[2],
                                             shrink_long_edges_xy_list[2]]).astype('int').reshape(-1, 1, 2)],
                                  True,
                                  (255, 255, 0), 2)
                # 左右短上下长的情况
                # --------------------------
                # |///|                 |///|
                # |///|                 |///|
                # |///|                 |///|
                # |///|                 |///|
                # ---------------------------
                elif first_long_edge_idx == 1:
                    # 左端阴影
                    cv2.polylines(draw_gt_quad_image,
                                  [np.array([xy_list[0], xy_list[1], shrink_long_edges_xy_list[1],
                                             shrink_long_edges_xy_list[0]]).astype('int').reshape(-1, 1, 2)],
                                  True,
                                  (255, 255, 0), 2)
                    # 右端阴影
                    cv2.polylines(draw_gt_quad_image,
                                  [np.array([shrink_long_edges_xy_list[3], shrink_long_edges_xy_list[2],
                                             xy_list[2], xy_list[3]]).astype('int').reshape(-1, 1, 2)],
                                  True,
                                  (255, 255, 0), 2)
        if config.SAVE_RESIZED_IMAGE:
            resized_image_filename = origin_image_filename
            resized_image_filepath = os.path.join(resized_image_dir, resized_image_filename)
            # cv2.namedWindow('resized_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('resized_image', image)
            # cv2.waitKey(0)
            cv2.imwrite(resized_image_filepath, image)
        if config.SAVE_RESIZED_LABEL:
            resized_label_filename = origin_label_filename[:-4] + '.npy'
            resized_label_filepath = os.path.join(resized_label_dir, resized_label_filename)
            np.save(resized_label_filepath, xy_list_array)
        if config.SAVE_DRAW_GT_QUAD_IMAGE:
            draw_gt_quad_image_filepath = os.path.join(draw_gt_quad_image_dir, origin_image_filename)
            # cv2.namedWindow('draw_gt_quad_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('draw_gt_quad_image', draw_gt_quad_image)
            # cv2.waitKey(0)
            cv2.imwrite(draw_gt_quad_image_filepath, draw_gt_quad_image)
        logger.debug('Handling {} ends'.format(origin_image_filepath))

    if config.SAVE_TRAIN_VAL_IMAGE:
        resized_image_dir = osp.join(dataset_dir, config.RESIZED_IMAGE_DIR_NAME)
        resized_label_dir = osp.join(dataset_dir, config.RESIZED_LABEL_DIR_NAME)
        for dir_type, dir_path in zip(['resized_image_dir', 'resized_label_dir'], [resized_image_dir, resized_label_dir]):
            if not osp.exists(dir_path) or len(os.listdir(dir_path)) == 0:
                logger.error('{}:{} does not exist or is empty'.format(dir_type, dir_path))
                return
        if not every_image_has_corresponding_label(resized_image_dir, resized_label_dir):
            logger.error('Files in resized_image_dir does not match files in resized_label_dir')
            return
        resized_image_filenames = os.listdir(resized_image_dir)
        num_resized_images = len(resized_image_filenames)
        # 存放用于 train 的 image
        train_image_dir = os.path.join(dataset_dir, config.TRAIN_IMAGE_DIR_NAME)
        # 存放用于 validation 的 image
        val_image_dir = os.path.join(dataset_dir, config.VAL_IMAGE_DIR_NAME)
        for dir_type, dir_path in zip(
                ['train_image_dir', 'val_image_dir'],
                [train_image_dir, val_image_dir]):
            if not os.path.exists(dir_path):
                logger.info('{}:{} does not exist, then creating'.format(dir_type, dir_path))
                os.mkdir(dir_path)
            else:
                logger.info('{}:{} already exists, then deleting and creating'.format(dir_type, dir_path))
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
        num_val_images = int(config.VAL_SPLIT_RATIO * num_resized_images)
        logger.debug('Expecting {} val images'.format(num_val_images))
        num_train_images = num_resized_images - num_val_images
        logger.debug('Expecting {} train images'.format(num_train_images))
        for image_filename in resized_image_filenames[:num_train_images]:
            train_image_filepath = osp.join(train_image_dir, image_filename)
            resized_image_filepath = osp.join(resized_image_dir, image_filename)
            shutil.copy(resized_image_filepath, train_image_filepath)
        for image_filename in resized_image_filenames[num_train_images:]:
            val_image_filepath = osp.join(val_image_dir, image_filename)
            resized_image_filepath = osp.join(resized_image_dir, image_filename)
            shutil.copy(resized_image_filepath, val_image_filepath)
        logger.info('Found {} train images.'.format(len(os.listdir(train_image_dir))))
        logger.info('Found {} val images.'.format(len(os.listdir(val_image_dir))))


if __name__ == '__main__':
    preprocess()
