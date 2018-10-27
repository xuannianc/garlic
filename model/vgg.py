from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization
import config


class East:
    def __init__(self):
        self.input = Input(name='input',
                           shape=(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3),
                           dtype='float32')
        vgg16 = VGG16(input_tensor=self.input,
                      weights='imagenet',
                      include_top=False)
        if config.LOCK_LAYERS:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        # feature maps [f1, f2, f3, f4]
        # 分别对应 vgg 中的 [block5_pool.output, block4_pool.output, block3_pool.output, block2_pool.output]
        self.f = [vgg16.get_layer('block{}_pool'.format(i + 1)).output for i in config.FEATURE_LAYERS_RANGE]
        # [None, f1, f2, f3, f4]
        self.f.insert(0, None)

    def g(self, i):
        """
        按照 paper 的架构图 g1=unpool(h1),g2=unpool(h2),g3=unpool(h3),g4=conv(h4)
        :param i:
        :return:
        """
        # i in cfg.FEATURE_LAYERS_RANGE
        assert i in config.FEATURE_LAYERS_RANGE, 'i={} not in {}'.format(i, config.FEATURE_LAYERS_RANGE)
        # g4 = conv3(h4)
        if i == config.FEATURE_LAYERS_RANGE[0]:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i in cfg.FEATURE_LAYERS_RANGE
        assert i in config.FEATURE_LAYERS_RANGE, 'i={} not in {}'.format(i, config.FEATURE_LAYERS_RANGE)
        # h1 == f1
        if i == 1:
            return self.f[i]
        # h2 == conv3(conv1(concat(g1,f2)))
        # h3 == conv3(conv1(concat(g2,f3)))
        # h4 == conv3(conv1(concat(g3,f4)))
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1, activation='relu', padding='same', )(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3, activation='relu', padding='same', )(bn2)
            return conv_3

    def build(self):
        # g4 output, shape (256, 256, 32)
        merge_output = self.g(config.FEATURE_LAYERS_RANGE[0])
        # (256, 256, 1)
        inside_shrink_quad_score = Conv2D(1, 1, padding='same', name='inside_shrink_quad_score')(merge_output)
        # (256, 256, 2)
        inside_end_quads_score = Conv2D(2, 1, padding='same', name='inside_end_quads_score')(merge_output)
        # (256, 256, 4)
        vertices_coord = Conv2D(4, 1, padding='same', name='vertices_coord')(merge_output)
        # (256, 256, 7)
        output = Concatenate(axis=-1, name='output')([inside_shrink_quad_score, inside_end_quads_score, vertices_coord])
        return Model(inputs=self.input, outputs=output)


# if __name__ == '__main__':
#     east = East()
#     east_model = east.build()
#     east_model.summary()
