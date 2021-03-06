import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


class DCNv2(layers.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 distribution='normal',
                 name='DCNv2'):
        super().__init__(name=name)
        assert distribution in ['uniform', 'normal']
        self.distribution = distribution
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv_offset_padding = layers.ZeroPadding2D(padding=((1, 0), (1,
                                                                          0)))
        self.zero_padding = layers.ZeroPadding2D(padding=((padding,
                                                           padding + 1),
                                                          (padding,
                                                           padding + 1)))

    def build(self, input_shape):
        input_dim = self.input_dim
        filters = self.filters
        filter_size = self.filter_size
        bias = self.bias
        self.offset_w = self.add_weight('offset_w',
                                        shape=[
                                            filter_size, filter_size,
                                            input_dim,
                                            filter_size * filter_size * 3
                                        ],
                                        initializer='zeros')
        self.offset_b = self.add_weight(
            'offset_b',
            shape=[1, 1, 1, filter_size * filter_size * 3],
            initializer='zeros')
        self.dcn_weight = self.add_weight(
            'dcn_weight',
            shape=[filters, input_dim, filter_size, filter_size],
            initializer=self.distribution)
        self.dcn_bias = None
        if bias:
            self.dcn_bias = self.add_weight('dcn_bias',
                                            shape=[
                                                filters,
                                            ],
                                            initializer='zeros')
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        filters = self.filters
        return (None, None, None, filters)

    def call(self, x, training=False):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        # ???filter_size = 3, stride = 2, padding = 1?????? ??????padding2 = 'valid'???K.conv2d???????????????self.conv_offset_padding
        # ???filter_size = 3, stride = 1, padding = 1?????? ??????padding2 = 'same'???K.conv2d?????????????????????self.conv_offset_padding
        # ?????????????????????self.zero_padding???????????????????????????
        if stride == 2:
            temp = self.conv_offset_padding(x)
        else:
            temp = x
        padding2 = None
        if stride == 2:
            padding2 = 'valid'
        else:
            padding2 = 'same'
        offset_mask = K.conv2d(temp,
                               self.offset_w,
                               strides=(stride, stride),
                               padding=padding2)
        offset_mask += self.offset_b

        offset_mask = tf.transpose(offset_mask, [0, 3, 1, 2])
        offset = offset_mask[:, :filter_size**2 * 2, :, :]
        mask = offset_mask[:, filter_size**2 * 2:, :, :]
        mask = tf.nn.sigmoid(mask)

        # ===================================
        N = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        out_C = tf.shape(dcn_weight)[0]
        in_C = tf.shape(dcn_weight)[1]
        kH = tf.shape(dcn_weight)[2]
        kW = tf.shape(dcn_weight)[3]
        W_f = tf.cast(W, tf.float32)
        H_f = tf.cast(H, tf.float32)
        kW_f = tf.cast(kW, tf.float32)
        kH_f = tf.cast(kH, tf.float32)

        out_W = (W_f + 2 * padding - (kW_f - 1)) // stride
        out_H = (H_f + 2 * padding - (kH_f - 1)) // stride
        out_W = tf.cast(out_W, tf.int32)
        out_H = tf.cast(out_H, tf.int32)
        out_W_f = tf.cast(out_W, tf.float32)
        out_H_f = tf.cast(out_H, tf.float32)

        # 1.????????????x??????????????????????????????pad_x
        pad_x = self.zero_padding(x)
        pad_x = tf.transpose(pad_x, [0, 3, 1, 2])

        # ?????????????????????pad_x????????????
        rows = tf.range(out_W_f, dtype=tf.float32) * stride + padding
        cols = tf.range(out_H_f, dtype=tf.float32) * stride + padding
        rows = tf.tile(rows[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis],
                       [1, out_H, 1, 1, 1])
        cols = tf.tile(cols[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis],
                       [1, 1, out_W, 1, 1])
        start_pos_yx = tf.concat(
            [cols, rows],
            axis=-1)  # [1, out_H, out_W, 1, 2]   ??????????????????????????????pad_x????????????
        start_pos_yx = tf.tile(
            start_pos_yx,
            [N, 1, 1, kH * kW, 1
             ])  # [N, out_H, out_W, kH*kW, 2]   ??????????????????????????????pad_x????????????
        start_pos_y = start_pos_yx[:, :, :, :, :
                                   1]  # [N, out_H, out_W, kH*kW, 1]   ??????????????????????????????pad_x????????????
        start_pos_x = start_pos_yx[:, :, :, :,
                                   1:]  # [N, out_H, out_W, kH*kW, 1]   ??????????????????????????????pad_x????????????

        # ????????????????????????
        half_W = (kW_f - 1) / 2
        half_H = (kH_f - 1) / 2
        rows2 = tf.range(kW_f, dtype=tf.float32) - half_W
        cols2 = tf.range(kH_f, dtype=tf.float32) - half_H
        rows2 = tf.tile(rows2[tf.newaxis, :, tf.newaxis], [kH, 1, 1])
        cols2 = tf.tile(cols2[:, tf.newaxis, tf.newaxis], [1, kW, 1])
        filter_inner_offset_yx = tf.concat([cols2, rows2],
                                           axis=-1)  # [kH, kW, 2]   ????????????????????????
        filter_inner_offset_yx = tf.reshape(
            filter_inner_offset_yx,
            (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   ????????????????????????
        filter_inner_offset_yx = tf.tile(
            filter_inner_offset_yx,
            [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   ????????????????????????
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :
                                                       1]  # [N, out_H, out_W, kH*kW, 1]   ????????????????????????
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :,
                                                       1:]  # [N, out_H, out_W, kH*kW, 1]   ????????????????????????

        mask = tf.transpose(mask, [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*1]
        offset = tf.transpose(offset,
                              [0, 2, 3, 1])  # [N, out_H, out_W, kH*kW*2]
        offset_yx = tf.reshape(
            offset,
            (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # ????????????
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = tf.maximum(pos_y, 0.0)
        pos_y = tf.minimum(pos_y, H_f + padding * 2 - 1.0)
        pos_x = tf.maximum(pos_x, 0.0)
        pos_x = tf.minimum(pos_x, W_f + padding * 2 - 1.0)
        ytxt = tf.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = tf.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]

        mask = tf.reshape(
            mask, (N, out_H, out_W, kH, kW))  # [N, out_H, out_W, kH, kW]

        def _process_sample(args):
            _pad_x, _mask, _ytxt = args
            # _pad_x:    [pad_x_H, pad_x_W, in_C]
            # _mask:     [out_H, out_W, kH, kW]
            # _ytxt:     [out_H, out_W, kH*kW, 2]

            _ytxt = tf.reshape(
                _ytxt, (out_H * out_W * kH * kW, 2))  # [out_H*out_W*kH*kW, 2]
            _yt = _ytxt[:, :1]
            _xt = _ytxt[:, 1:]
            _y1 = tf.floor(_yt)
            _x1 = tf.floor(_xt)
            _y2 = _y1 + 1.0
            _x2 = _x1 + 1.0
            _y1x1 = tf.concat([_y1, _x1], -1)
            _y1x2 = tf.concat([_y1, _x2], -1)
            _y2x1 = tf.concat([_y2, _x1], -1)
            _y2x2 = tf.concat([_y2, _x2], -1)

            _y1x1_int = tf.cast(_y1x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v1 = tf.gather_nd(_pad_x, _y1x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y1x2_int = tf.cast(_y1x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v2 = tf.gather_nd(_pad_x, _y1x2_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x1_int = tf.cast(_y2x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v3 = tf.gather_nd(_pad_x, _y2x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x2_int = tf.cast(_y2x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v4 = tf.gather_nd(_pad_x, _y2x2_int)  # [out_H*out_W*kH*kW, in_C]

            lh = _yt - _y1  # [out_H*out_W*kH*kW, 1]
            lw = _xt - _x1
            hh = 1 - lh
            hw = 1 - lw
            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw
            value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [out_H*out_W*kH*kW, in_C]
            _mask = tf.reshape(_mask, (out_H * out_W * kH * kW, 1))
            value = value * _mask
            value = tf.reshape(value, (out_H, out_W, kH, kW, in_C))
            value = tf.transpose(
                value, [0, 1, 4, 2, 3])  # [out_H, out_W, in_C, kH, kW]
            return value

        # ???????????????????????????1x1????????????????????????????????????
        new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt],
                          dtype=tf.float32)  # [N, out_H, out_W, in_C, kH, kW]
        new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH *
                                   kW))  # [N, out_H, out_W, in_C * kH * kW]
        tw = tf.transpose(
            dcn_weight,
            [1, 2, 3, 0])  # [out_C, in_C, kH, kW] -> [in_C, kH, kW, out_C]
        tw = tf.reshape(tw, (1, 1, in_C * kH * kW,
                             out_C))  # [1, 1, in_C*kH*kW, out_C]  ??????1x1?????????
        out = K.conv2d(new_x, tw, strides=(1, 1),
                       padding='valid')  # [N, out_H, out_W, out_C]
        return out


if __name__ == "__main__":
    model = DCNv2(input_dim=12, filters=12, filter_size=3, stride=1, padding=1)
    model.build((None, None, None))
    K.batch_set_value(tuples=[
        (model.dcn_weight, tf.ones(model.dcn_weight.shape)),
    ])
    x = tf.reshape(tf.range(10 * 10 * 12 * 2), (2, 10, 10, 12))
    x = tf.cast(x, tf.float32)
    out = model(x)
    print(tf.squeeze(out))