import tensorflow as tf
import config
from keras import backend as K


def loss(y_true, y_pred):
    """
    loss for inside_shrink_quad_score, inside_end_quads_score, vertices_coord
    :param y_true:
    :param y_pred:
    :return:
    """
    ######################################## inside_shrink_quad_score_loss #############################
    labels1 = y_true[:, :, :, :1]
    logits1 = y_pred[:, :, :, :1]
    # apply sigmoid activation
    predicts1 = tf.nn.sigmoid(logits1)
    # balance positive and negative samples in an image
    # gt 中的值为 0 (negative samples) 的比例
    beta1 = 1 - tf.reduce_mean(labels1)
    # log +epsilon for stable cal
    inside_shrink_quad_score_loss = tf.reduce_mean(
        -1 * (beta1 * labels1 * tf.log(predicts1 + config.EPSILON) +
              (1 - beta1) * (1 - labels1) * tf.log(1 - predicts1 + config.EPSILON)))
    inside_shrink_quad_score_loss *= config.LAMBDA_INSIDE_SHRINK_QUAD_SCORE_LOSS

    ######################################## inside_end_quad_score_loss ################################
    labels2 = y_true[:, :, :, 1:3]
    logits2 = y_pred[:, :, :, 1:3]
    predicts2 = tf.nn.sigmoid(logits2)
    # 预测是否在 end quad 的 loss
    # 在 shrink quad 但是不在 end quads 的比例
    beta2 = 1 - (tf.reduce_mean(labels2[:, :, :, 1]) / (tf.reduce_mean(labels1) + config.EPSILON))
    pos = -1 * beta2 * labels2 * tf.log(predicts2 + config.EPSILON)
    neg = (1 - beta2) * (1 - labels2) * tf.log(1 - predicts2 + config.EPSILON)
    inside_shrink_quad_mask = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
    inside_end_quads_score_loss_part1 = \
        tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * inside_shrink_quad_mask) / (
                tf.reduce_sum(inside_shrink_quad_mask) + config.EPSILON)
    # 预测是哪一个 end quad 的 loss
    inside_end_quads_mask = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)
    inside_end_quads_score_loss_part2 = K.sum(
        inside_end_quads_mask * K.binary_crossentropy(y_true[:, :, :, 2], y_pred[:, :, :, 2])) / K.sum(
        inside_end_quads_mask + config.EPSILON)
    inside_end_quads_score_loss = inside_end_quads_score_loss_part1 + inside_end_quads_score_loss_part2
    inside_end_quads_score_loss *= config.LAMBDA_INSIDE_END_QUADS_SCORE_LOSS

    ####################################### vertices_coord_loss ###############################################
    labels3 = y_true[:, :, :, 3:]
    predicts3 = y_pred[:, :, :, 3:]
    loss = smooth_l1_loss(labels3, predicts3, inside_end_quads_mask)
    vertices_coord_loss = tf.reduce_sum(loss) / (tf.reduce_sum(inside_end_quads_mask) + config.EPSILON)
    vertices_coord_loss *= config.LAMBDA_VERTICES_COORD_LOSS
    return inside_shrink_quad_score_loss + inside_end_quads_score_loss + vertices_coord_loss


def smooth_l1_loss(y_true, y_pred, mask=None):
    """Implements Smooth-L1 loss.
    y_true and y_pred are : (256, 256, 4)

    Return:
        loss: shape 为 (256, 256)
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    # (256, 256, 4)
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    # (256, 256)
    loss = K.mean(loss, axis=-1)
    if mask is not None:
        loss = mask * loss
    return loss