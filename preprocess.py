import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm

import cfg
from label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
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
               / (xy_list[other_vertex_idx, 0] - xy_list[first_vertex_idx, 0] + cfg.epsilon)
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
            xy_list[second_vertex_idx, 0] - xy_list[fourth_vertex_idx, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            show_gt_im = im.copy()
            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir, o_img_fname[:-4] + '.txt'), 'r') as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                xy_list_array[i] = xy_list
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_long_edges_xy_list, _, long_edge_idx = shrink(xy_list, cfg.shrink_side_ratio)
                if draw_gt_quad:
                    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill='green')
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill='blue')
                    # 这里如果分开讨论更容易理解
                    ############################### 原来的实现 #################################
                    # vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                    #       [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    # for q_th in range(2):
                    #     draw.line([tuple(xy_list[vs[long_edge_idx][q_th][0]]),
                    #                tuple(shrink_long_edges_xy_list[vs[long_edge_idx][q_th][1]]),
                    #                tuple(shrink_long_edges_xy_list[vs[long_edge_idx][q_th][2]]),
                    #                tuple(xy_list[vs[long_edge_idx][q_th][3]]),
                    #                tuple(xy_list[vs[long_edge_idx][q_th][4]])],
                    #               width=3, fill='yellow')
                    ############################### FIXME: adam 的实现 ###############################
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
                    if long_edge_idx == 0:
                        # 上端阴影
                        draw.line([tuple(xy_list[0]),
                                   tuple(shrink_long_edges_xy_list[0]),
                                   tuple(shrink_long_edges_xy_list[3]),
                                   tuple(xy_list[3]),
                                   tuple(xy_list[0])], width=3, fill='yellow')
                        # 下端阴影
                        draw.line([tuple(shrink_long_edges_xy_list[1]),
                                   tuple(xy_list[1]),
                                   tuple(xy_list[2]),
                                   tuple(shrink_long_edges_xy_list[2]),
                                   tuple(shrink_long_edges_xy_list[1])], width=3, fill='yellow')
                    # 左右短上下长的情况
                    # --------------------------
                    # |///|                 |///|
                    # |///|                 |///|
                    # |///|                 |///|
                    # |///|                 |///|
                    # ---------------------------
                    elif long_edge_idx == 1:
                        # 左端阴影
                        draw.line([tuple(xy_list[0]),
                                   tuple(xy_list[1]),
                                   tuple(shrink_long_edges_xy_list[1]),
                                   tuple(shrink_long_edges_xy_list[0]),
                                   tuple(xy_list[0])], width=3, fill='yellow')
                        # 右端阴影
                        draw.line([tuple(shrink_long_edges_xy_list[3]),
                                   tuple(shrink_long_edges_xy_list[2]),
                                   tuple(xy_list[2]),
                                   tuple(xy_list[3]),
                                   tuple(shrink_long_edges_xy_list[3])], width=3, fill='yellow')
            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    preprocess()
