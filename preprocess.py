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

logger = logging.getLogger('preprocess')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


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
    # reorder_xy_list 的 shape 为 (4, 2)
    reorder_xy_list = np.zeros_like(xy_list)
    ############################### 找最小 x 的点,其下标复制给 first_v ########################################
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    # 四个点的坐标按 x 进行排序, 如果 x 相等, 按 y 排序
    # np.argsort 返回结果的 shape 和原来一样, 每个元素的值表示该位置对应的原数组中的元素的序号
    # 参见 https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argsort.html
    ordered_idxes = np.argsort(xy_list, axis=0)
    # 有最小 x 的元素在原数组中的序号
    xmin1_idx = ordered_idxes[0, 0]
    # 有倒数第二小的 x 的元素在原数组的序号
    xmin2_idx = ordered_idxes[1, 0]
    # 如果最小的两个 x 相等, 比较 y
    if xy_list[xmin1_idx, 0] == xy_list[xmin2_idx, 0]:
        if xy_list[xmin1_idx, 1] <= xy_list[xmin2_idx, 1]:
            reorder_xy_list[0] = xy_list[xmin1_idx]
            first_vertex_idx = xmin1_idx
        else:
            reorder_xy_list[0] = xy_list[xmin2_idx]
            first_vertex_idx = xmin2_idx
    else:
        # 把有最小 x 的元素放在  reorder_xy_list 的下标 0 的位置
        reorder_xy_list[0] = xy_list[xmin1_idx]
        first_vertex_idx = xmin1_idx
    ##################################### 找到第一个顶点的对角线的点　############################################
    # connect the first point to the third point on the other side of
    # the line with the middle slope
    other_vertex_idxes = list(range(4))
    other_vertex_idxes.remove(first_vertex_idx)
    # k 存放第一个点和其他点的斜率
    k = np.zeros((len(other_vertex_idxes),))
    for i, other_vertex_idx in enumerate(other_vertex_idxes):
        k[i] = (xy_list[other_vertex_idx, 1] - xy_list[first_vertex_idx, 1]) \
               / (xy_list[other_vertex_idx, 0] - xy_list[first_vertex_idx, 0] + config.EPSILON)
    # ｋ_mid_idx 是三个斜率的中间值的下标
    k_mid_idx = np.argsort(k)[1]
    k_mid = k[k_mid_idx]
    third_vertex_idx = other_vertex_idxes[k_mid_idx]
    reorder_xy_list[2] = xy_list[third_vertex_idx]
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
    reorder_xy_list[1] = xy_list[second_vertex_idx]
    reorder_xy_list[3] = xy_list[fourth_vertex_idx]
    ############################### 把左上方的点作为第一个点,按逆时针得到其他点　######################################
    # compare slope of 13 and 24, determine the final order
    k13 = k_mid
    k24 = (xy_list[second_vertex_idx, 1] - xy_list[fourth_vertex_idx, 1]) / (
            xy_list[second_vertex_idx, 0] - xy_list[fourth_vertex_idx, 0] + config.EPSILON)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


# def resize_image(im, max_img_size=cfg.max_train_img_size):
#     im_width = np.minimum(im.width, max_img_size)
#     if im_width == max_img_size < im.width:
#         im_height = int((im_width / im.width) * im.height)
#     else:
#         im_height = im.height
#     o_height = np.minimum(im_height, max_img_size)
#     if o_height == max_img_size < im_height:
#         o_width = int((o_height / im_height) * im_width)
#     else:
#         o_width = im_width
#     d_wight = o_width - (o_width % 32)
#     d_height = o_height - (o_height % 32)
#     return d_wight, d_height


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
        image = cv2.resize(image, ((round(w * scale), round(h * scale))), interpolation=cv2.INTER_LINEAR)
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


def preprocess():
    dataset_dir = config.DATASET_DIR
    # 存放原来的图片
    origin_image_dir = os.path.join(dataset_dir, config.ORIGIN_IMAGE_DIR_NAME)
    # 存放原来的标注
    origin_label_dir = os.path.join(dataset_dir, config.ORIGIN_LABEL_DIR_NAME)
    # 存放 resize 后的图片
    current_image_dir = os.path.join(dataset_dir, config.CURRENT_IMAGE_DIR_NAME)
    # 存放 resize 后的标注
    current_label_dir = os.path.join(dataset_dir, config.CURRENT_LABEL_DIR_NAME)
    if not os.path.exists(current_image_dir):
        os.mkdir(current_image_dir)
    if not os.path.exists(current_label_dir):
        os.mkdir(current_label_dir)
    draw_gt_quad_image_dir = os.path.join(dataset_dir, config.DRAW_GT_QUAD_IMAGE_DIR_NAME)
    if not os.path.exists(draw_gt_quad_image_dir):
        os.mkdir(draw_gt_quad_image_dir)

    origin_image_filenames = os.listdir(origin_image_dir)
    num_origin_images = len(origin_image_filenames)
    logger.info('Found {} origin images.'.format(num_origin_images))
    train_val_set = []
    for origin_image_filename, _ in zip(origin_image_filenames, tqdm(range(num_origin_images))):
        origin_image_filepath = os.path.join(origin_image_dir, origin_image_filename)
        logger.debug('Handling {} starts'.format(origin_image_filepath))
        origin_image = cv2.imread(origin_image_filepath)
        cv2.namedWindow('origin_image', cv2.WINDOW_NORMAL)
        cv2.imshow('origin_image', origin_image)
        cv2.waitKey(0)
        image, window, scale, padding, crop = resize_image(origin_image,
                                                           min_dim=config.IMAGE_MIN_DIM,
                                                           max_dim=config.IMAGE_MAX_DIM,
                                                           mode=config.IMAGE_RESIZE_MODE)
        cv2.namedWindow('current_image', cv2.WINDOW_NORMAL)
        cv2.imshow('current_image', image)
        cv2.waitKey(0)
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
                cv2.polylines(draw_gt_quad_image, [xy_list.astype('int').reshape(-1, 1, 2)], True, (0, 255, 0), 1)
                cv2.polylines(draw_gt_quad_image, [shrink_xy_list.astype('int').reshape(-1, 1, 2)], True, (255, 0, 0), 1)
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
            current_image_filename = origin_image_filename
            current_image_filepath = os.path.join(current_image_dir, current_image_filename)
            # cv2.imwrite(current_image_filepath, image)
        current_label_filename = origin_label_filename[:-4] + '.npy'
        current_label_filepath = os.path.join(current_label_dir, current_label_filename)
        np.save(current_label_filepath, xy_list_array)
        if config.DRAW_GT_QUAD:
            draw_gt_quad_image_filepath = os.path.join(draw_gt_quad_image_dir, current_image_filename)
            cv2.namedWindow('draw_gt_quad_image', cv2.WINDOW_NORMAL)
            cv2.imshow('draw_gt_quad_image', draw_gt_quad_image)
            cv2.waitKey(0)
            # cv2.imwrite(draw_gt_quad_image_filepath, draw_gt_quad_image)
        train_val_set.append('{},{},{},{},{}\n'.format(current_image_filepath,
                                                       list(origin_image.shape[:2]),
                                                       list(image.shape[:2]),
                                                       list(window),
                                                       scale))
        logger.debug('Handling {} ends'.format(origin_image_filepath))

    current_image_filenames = os.listdir(current_image_dir)
    num_current_images = len(current_image_filenames)
    print('Found {} current images.'.format(num_current_images))
    current_label_filenames = os.listdir(current_label_dir)
    num_current_labels = len(current_label_filenames)
    print('Found {} current labels.'.format(num_current_labels))

    random.shuffle(train_val_set)
    val_count = int(config.validation_split_ratio * len(train_val_set))
    with open(os.path.join(dataset_dir, config.VAL_FILENAME), 'w') as val_file:
        val_file.writelines(train_val_set[:val_count])
    with open(os.path.join(dataset_dir, config.TRAIN_FILENAME), 'w') as train_file:
        train_file.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    preprocess()
