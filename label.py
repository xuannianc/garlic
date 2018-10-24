import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import config


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


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


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


def process_label(dataset_dir=config.DATASET_DIR):
    with open(os.path.join(dataset_dir, config.VAL_FILENAME), 'r') as val_file:
        data_lines = val_file.readlines()
    with open(os.path.join(dataset_dir, config.TRAIN_FILENAME), 'r') as train_file:
        data_lines.extend(train_file.readlines())
    for line, _ in zip(data_lines, tqdm(range(len(data_lines)))):
        line_cols = str(line).strip().split(',')
        img_name, width, height = line_cols[0].strip(), int(line_cols[1].strip()), int(line_cols[2].strip())
        gt = np.zeros((height // config.pixel_size, width // config.pixel_size, 7))
        train_label_dir = os.path.join(dataset_dir, config.train_label_dir_name)
        xy_list_array = np.load(os.path.join(train_label_dir, img_name[:-4] + '.npy'))
        train_image_dir = os.path.join(dataset_dir, config.train_image_dir_name)
        with Image.open(os.path.join(train_image_dir, img_name)) as im:
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, config.SHRINK_RATIO)
                shrink_1, _, long_edge = shrink(xy_list, config.SHRINK_SIDE_RATIO)
                # shape 为　(2,), 第一个元素为 min_x, 第二个元素为　min_y
                float_min_xy = np.amin(shrink_xy_list, axis=0)
                # shape 为　(2,), 第一个元素为 max_x, 第二个元素为　max_y
                float_max_xy = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                int_min_xy = np.floor(float_min_xy / config.PIXEL_SIZE).astype('int')
                # ceil of the float
                int_max_xy = np.ceil(float_max_xy / config.PIXEL_SIZE).astype('int')
                int_min_y = np.maximum(0, int_min_xy[1])
                int_max_y = np.minimum(height // config.PIXEL_SIZE, int_max_xy[1])
                int_min_x = np.maximum(0, int_min_xy[0])
                int_max_x = np.minimum(width // config.PIXEL_SIZE, int_max_xy[0])
                for y in range(int_min_y, int_max_y):
                    for x in range(int_min_x, int_max_x):
                        px = x * config.PIXEL_SIZE
                        py = y * config.PIXEL_SIZE
                        if point_inside_of_quad(px, py, shrink_xy_list, float_min_xy, float_max_xy):
                            gt[y, x, 0] = 1
                            line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[y, x, 1] = 1
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[y, x, 2:3] = ith
                                gt[y, x, 3:5] = \
                                    xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[y, x, 5:] = \
                                    xy_list[vs[long_edge][ith][1]] - [px, py]
                            draw.line([(px - 0.5 * config.pixel_size,
                                        py - 0.5 * config.pixel_size),
                                       (px + 0.5 * config.pixel_size,
                                        py - 0.5 * config.pixel_size),
                                       (px + 0.5 * config.pixel_size,
                                        py + 0.5 * config.pixel_size),
                                       (px - 0.5 * config.pixel_size,
                                        py + 0.5 * config.pixel_size),
                                       (px - 0.5 * config.pixel_size,
                                        py - 0.5 * config.pixel_size)],
                                      width=line_width, fill=line_color)
            act_image_dir = os.path.join(config.data_dir, config.show_act_image_dir_name)
            if config.draw_act_quad:
                im.save(os.path.join(act_image_dir, img_name))
        train_label_dir = os.path.join(dataset_dir, config.train_label_dir_name)
        np.save(os.path.join(train_label_dir, img_name[:-4] + '_gt.npy'), gt)


if __name__ == '__main__':
    process_label()
