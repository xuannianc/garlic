import numpy as np
import os
from tqdm import tqdm
import config
import logging
import sys
import cv2
import os.path as osp
import shutil

logger = logging.getLogger('label')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        # array([[x1-x0,y1-y0],[x2-x1,y2-y1],[x3-x2,y3-y2],[0,0]]
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        # array([[x1-x0,y1-y0],[x2-x1,y2-y1],[x3-x2,y3-y2],[x0-x3,y0-y3]])
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        # array([[y0,x0],[y1,x1],[y2,x2],[y3,x3]])
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        # array([[(x1-x0)*(py-y0),(y1-y0)*(px-x0)],[(x2-x1)*(py-y1),(y2-y1)*(px-x1)]...)
        a = xy_list * ([py, px] - yx_list)
        # array([(x1-x0)*(py-y0)-(y1-y0)*(px-x0),(x2-x1)*(py-y1)-(y2-y1)*(px-x1)]...)
        b = a[:, 0] - a[:, 1]
        # 非常神奇
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_end_quads(center_x, center_y, xy_list, shrink_long_edges_xy_list, first_long_edge_idx):
    """
    判断中心点是否在两端的 quad 中
    :param center_x:
    :param center_y:
    :param xy_list:
    :param shrink_long_edges_xy_list:
    :param first_long_edge_idx:
    :return: 如果不在,返回 -1;如果在上端或者左端返回 0; 如果在下端或者右端返回 1
    """
    idx = -1
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
        up_quad_xy_list = np.array([xy_list[0], shrink_long_edges_xy_list[0], shrink_long_edges_xy_list[3], xy_list[3]])
        # 下端阴影
        down_quad_xy_list = np.array(
            [shrink_long_edges_xy_list[1], xy_list[1], xy_list[2], shrink_long_edges_xy_list[2]])
        up_float_min_xy = np.amin(up_quad_xy_list, axis=0)
        up_float_max_xy = np.amax(up_quad_xy_list, axis=0)
        down_float_min_xy = np.amin(down_quad_xy_list, axis=0)
        down_float_max_xy = np.amax(down_quad_xy_list, axis=0)
        if point_inside_of_quad(center_x, center_y, up_quad_xy_list, up_float_min_xy, up_float_max_xy):
            idx = 0
        if point_inside_of_quad(center_x, center_y, down_quad_xy_list, down_float_min_xy, down_float_max_xy):
            if idx == 0:
                # logger.warning('center point in both end quads')
                idx = -1
            else:
                idx = 1
    # 左右短上下长的情况
    # --------------------------
    # |///|                 |///|
    # |///|                 |///|
    # |///|                 |///|
    # |///|                 |///|
    # ---------------------------
    elif first_long_edge_idx == 1:
        # 左端阴影
        left_quad_xy_list = np.array(
            [xy_list[0], xy_list[1], shrink_long_edges_xy_list[1], shrink_long_edges_xy_list[0]])
        # 右端阴影
        right_quad_xy_list = np.array(
            [shrink_long_edges_xy_list[3], shrink_long_edges_xy_list[2], xy_list[2], xy_list[3]])
        left_float_min_xy = np.amin(left_quad_xy_list, axis=0)
        left_float_max_xy = np.amax(left_quad_xy_list, axis=0)
        right_float_min_xy = np.amin(right_quad_xy_list, axis=0)
        right_float_max_xy = np.amax(right_quad_xy_list, axis=0)
        if point_inside_of_quad(center_x, center_y, left_quad_xy_list, left_float_min_xy, left_float_max_xy):
            idx = 0
        if point_inside_of_quad(center_x, center_y, right_quad_xy_list, right_float_min_xy, right_float_max_xy):
            if idx == 0:
                # logger.warning('center point in both end quads')
                idx = -1
            else:
                idx = 1
    return idx


def shrink(xy_list, shrink_ratio=config.SHRINK_RATIO):
    """
    把 gt_quad(ground-truth quadrangle) 按照一定比例收缩
    :param xy_list: gt_quad 的四个点坐标, 是从左上角的点开始按逆时针旋转得到的四个点的坐标, shape 为 (4,2)
    :param shrink_ratio: 收缩比例
    :return:
    """
    if shrink_ratio == 0.0:
        return xy_list, xy_list
    # [[x0 - x1, y0 - y1], [x1 - x2, y1 - y2], [x2 - x3, y2 - y3]]
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    # [[x3 - x0, y3 - y0]]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    # [[x0 - x1, y0 - y1], [x1 - x2, y1 - y2], [x2 - x3, y2 - y3],[x3 - x0, y3 - y0]]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    # [sqrt((x0 - x1)^2 + (y0 - y1)^2), sqrt((x1 - x2)^2+(y1 - y2)^2),
    # sqrt((x2 - x3)^2+(y2 - y3)^2),sqrt((x3 - x0)^2+(y3 - y0)^2)]
    # 分别表示 [d01, d12, d23, d30]
    distances = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    # np.argmax([d01 + d23, d12 + d34]), 所有取值为 0 或者 1
    first_long_edge_idx = np.argmax(np.sum(np.reshape(distances, (2, 2)), axis=0))
    # 取值为 1 或者 0
    first_short_edge_idx = 1 - first_long_edge_idx
    second_long_edge_idx = first_long_edge_idx + 2
    second_short_edge_idx = first_short_edge_idx + 2
    # cal renference lengths array, refer to paper
    # 每一个元素表示该位置的顶点连接的较短的那条边的长度
    # [min(d01,d30),min(d12,d01),min(d23,d12),min(d30,d23)]
    r = [np.minimum(distances[i], distances[(i - 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += config.EPSILON
    # np.arctan() 是 tan() 的反操作,y = tan(x) 那么 x = argtan(y)
    # 同理 |delta_y/delta_x| = tan(theta), theta = arctan(|delta_y/delta_x|)
    # [arctan(|y0-y1/x0-x1|),arctan(|y1-y2/x1-x2|),arctan(|y2-y3/x2-x3|),arctan(|y3-y0/x3-x0|)|
    # thetas 表示每条边的斜率角度, shape 为 (4,)
    thetas = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, first_long_edge_idx, r, thetas, shrink_ratio)
    shrink_edge(xy_list, temp_new_xy_list, second_long_edge_idx, r, thetas, shrink_ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, first_short_edge_idx, r, thetas, shrink_ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, second_short_edge_idx, r, thetas, shrink_ratio)
    return temp_new_xy_list, new_xy_list, first_long_edge_idx


def shrink_edge(xy_list, new_xy_list, edge_idx, r, theta, shrink_ratio=config.SHRINK_RATIO):
    """
    收缩某条边
    :param xy_list: 收缩前的坐标, shape 为 (4,2)
    :param new_xy_list: 收缩后的坐标, shape 为 (4,2)
    :param edge_idx: 边的 index, int
    :param r: reference lengths, shape 为 (4,)
    :param theta: 每条边的斜率角度, shape 为 (4,)
    :param shrink_ratio: 收缩比例
    :return: None, 只是对 new_xy_list 中的值作更改
    """
    if shrink_ratio == 0.0:
        return
    start_vertex_idx = edge_idx
    end_vertex_idx = (edge_idx + 1) % 4
    # vertex_idx=0 delta_x_sign = -1
    #           ________
    #          /       /
    #         /       /
    #        /_______/
    # vertex_idx=0 delta_x_sign == 1
    #      _________
    #      \        \
    #       \        \
    #        \________\
    # 计算收缩后的 start_vertex_idx 的点的坐标
    delta_x_sign = np.sign(xy_list[end_vertex_idx, 0] - xy_list[start_vertex_idx, 0])
    new_xy_list[start_vertex_idx, 0] += delta_x_sign * shrink_ratio * r[start_vertex_idx] * np.cos(theta[edge_idx])
    delta_y_sign = np.sign(xy_list[end_vertex_idx, 1] - xy_list[start_vertex_idx, 1])
    new_xy_list[start_vertex_idx, 1] += delta_y_sign * shrink_ratio * r[start_vertex_idx] * np.sin(theta[edge_idx])
    # 计算收缩后的 end_vertex_idx 的点的坐标
    delta_x_sign = -1 * delta_x_sign
    new_xy_list[end_vertex_idx, 0] += delta_x_sign * shrink_ratio * r[end_vertex_idx] * np.cos(theta[edge_idx])
    delta_y_sign = -1 * delta_y_sign
    new_xy_list[end_vertex_idx, 1] += delta_y_sign * shrink_ratio * r[end_vertex_idx] * np.sin(theta[edge_idx])


def generate_label(dataset_dir=config.DATASET_DIR):
    """
    生成用于训练的 label
    :param dataset_dir:
    :return:
    """
    resized_label_dir = osp.join(dataset_dir, config.RESIZED_LABEL_DIR_NAME)
    train_image_dir = osp.join(dataset_dir, config.TRAIN_IMAGE_DIR_NAME)
    val_image_dir = osp.join(dataset_dir, config.VAL_IMAGE_DIR_NAME)
    for dir_type, dir_path in zip(['resized_label_dir, train_image_dir, val_image_dir'],
                                  [resized_label_dir, train_image_dir, val_image_dir]):
        if not osp.exists(dir_path) or len(os.listdir(dir_path)) == 0:
            logger.error('{}:{} does not exist or is empty'.format(dir_type, dir_path))
            return
    train_label_dir = osp.join(dataset_dir, config.TRAIN_LABEL_DIR_NAME)
    val_label_dir = osp.join(dataset_dir, config.VAL_LABEL_DIR_NAME)
    # 存放画有 act 的 resized_image
    draw_act_image_dir = osp.join(dataset_dir, config.DRAW_ACT_IMAGE_DIR_NAME)
    for dir_type, dir_path in zip(['train_label_dir', 'val_label_dir', 'draw_act_image_dir'],
                                  [train_label_dir, val_label_dir, draw_act_image_dir]):
        if not osp.exists(dir_path):
            logger.info('{}:{} does not exist, then creating'.format(dir_type, dir_path))
            os.mkdir(dir_path)
        else:
            logger.info('{}:{} already exists, then deleting and creating'.format(dir_type, dir_path))
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    image_dirs = [train_image_dir, val_image_dir]
    label_dirs = [train_label_dir, val_label_dir]
    for i, image_dir in enumerate(image_dirs):
        image_filenames = os.listdir(image_dir)
        num_images = len(image_filenames)
        for image_filename, _ in zip(image_filenames, tqdm(range(num_images))):
            image_filepath = osp.join(image_dirs[i], image_filename)
            logger.debug('Handling {} starts'.format(image_filepath))
            image = cv2.imread(image_filepath)
            # cv2.namedWindow('origin_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('origin_image', image)
            # cv2.waitKey(0)
            height, width = image.shape[:2]
            gt = np.zeros((height // config.PIXEL_SIZE, width // config.PIXEL_SIZE, 7))
            xy_list_array = np.load(osp.join(resized_label_dir, image_filename[:-4] + '.npy'))
            draw_act_image = image.copy()
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, config.SHRINK_RATIO)
                shrink_long_edges_xy_list, _, first_long_edge_idx = shrink(xy_list, config.SHRINK_SIDE_RATIO)
                # shape 为　(2,), 第一个元素为 min_x, 第二个元素为　min_y
                float_min_xy = np.amin(shrink_xy_list, axis=0)
                # shape 为　(2,), 第一个元素为 max_x, 第二个元素为　max_y
                float_max_xy = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                # 举例 float_min_x == 197.2 那么 int_min_x = 49, 表示在第 49 个矩形框中, 第 49 个矩形框的 center_x=49*4+2=198
                # 198 > 197.2 所以认为这个矩形框就是在四边形内
                # 同理如果 float_max_x == 197.2, 那么 int_max_x = 50, range 中最大能访问到 49, 第 49 个矩形框的 center_x=198
                # 198 > 197.2 所以认为这个矩形框不在四边形内
                int_min_xy = np.floor(float_min_xy / config.PIXEL_SIZE).astype('int')
                # ceil of the float
                int_max_xy = np.ceil(float_max_xy / config.PIXEL_SIZE).astype('int')
                int_min_y = np.maximum(0, int_min_xy[1])
                int_max_y = np.minimum(height // config.PIXEL_SIZE, int_max_xy[1])
                int_min_x = np.maximum(0, int_min_xy[0])
                int_max_x = np.minimum(width // config.PIXEL_SIZE, int_max_xy[0])
                for y in range(int_min_y, int_max_y):
                    for x in range(int_min_x, int_max_x):
                        #  -------
                        # |---x---|
                        #  -------
                        # PIXEL_SIZE * PIXEL_SIZE 矩形框中心点的坐标
                        center_x = (x + 0.5) * config.PIXEL_SIZE
                        center_y = (y + 0.5) * config.PIXEL_SIZE
                        # 判断中心点是否在四边形中
                        if point_inside_of_quad(center_x, center_y, shrink_xy_list, float_min_xy, float_max_xy):
                            # 如果在,设置与矩形框对应的 gt 为 1
                            gt[y, x, 0] = 1
                            line_width, line_color = 1, (0, 0, 255)
                            # 判断中心点是否在收缩矩形框的两端
                            idx = point_inside_of_end_quads(center_x, center_y, xy_list, shrink_long_edges_xy_list,
                                                            first_long_edge_idx)
                            if idx in range(2):
                                gt[y, x, 1] = 1
                                # 首端
                                if idx == 0:
                                    line_width, line_color = 2, (0, 255, 255)
                                    gt[y, x, 2] = idx
                                    # 上端
                                    if first_long_edge_idx == 0:
                                        # 按照逆时针, 先右上再左上
                                        gt[y, x, 3:5] = xy_list[3] - [center_x, center_y]
                                        gt[y, x, 5:] = xy_list[0] - [center_x, center_y]
                                    # 左端
                                    else:
                                        # 按照逆时针, 先左上后左下
                                        gt[y, x, 3:5] = xy_list[0] - [center_x, center_y]
                                        gt[y, x, 5:] = xy_list[1] - [center_x, center_y]
                                # 尾端
                                else:
                                    line_width, line_color = 2, (0, 255, 0)
                                    gt[y, x, 2] = idx
                                    # 下端
                                    if first_long_edge_idx == 0:
                                        # 按照逆时针, 先左下 (1) 后右下 (2)
                                        gt[y, x, 3:5] = xy_list[1] - [center_x, center_y]
                                        gt[y, x, 5:] = xy_list[2] - [center_x, center_y]
                                    # 右端
                                    else:
                                        # 按照逆时针, 先右下 (2) 后右上 (3)
                                        gt[y, x, 3:5] = xy_list[2] - [center_x, center_y]
                                        gt[y, x, 5:] = xy_list[3] - [center_x, center_y]
                            if config.DRAW_ACT:
                                points_around_center = np.array([[center_x - 2, center_y - 2],
                                                                 [center_x - 2, center_y + 2],
                                                                 [center_x + 2, center_y + 2],
                                                                 [center_x + 2, center_y - 2]]).astype('int')
                                cv2.polylines(draw_act_image, [points_around_center.reshape(-1, 1, 2)], True,
                                              line_color,
                                              line_width)
            # cv2.namedWindow('draw_act_image', cv2.WINDOW_NORMAL)
            # cv2.imshow('draw_act_image', draw_act_image)
            # cv2.waitKey(0)
            cv2.imwrite(osp.join(draw_act_image_dir, image_filename), draw_act_image)
            np.save(os.path.join(label_dirs[i], image_filename[:-4] + '_gt.npy'), gt)
            logger.debug('Handling {} ends'.format(image_filepath))


if __name__ == '__main__':
    generate_label()
