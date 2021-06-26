import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNv2(nn.Module):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 distribution='normal',
                 gain=1):
        super().__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = nn.Conv2d(input_dim,
                                     filter_size * filter_size * 3,
                                     kernel_size=filter_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=True)
        # 初始化代码摘抄自SOLOv2  mmcv/cnn/weight_init.py  里的代码
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

        self.sigmoid = nn.Sigmoid()

        self.dcn_weight = nn.Parameter(
            torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias:
            self.dcn_bias = nn.Parameter(torch.randn(filters, ))
            nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def gather_nd(self, input, index):
        # 不被合并的后面的维
        keep_dims = []
        # 被合并的维
        first_dims = []
        dim_idx = []
        dims = index.shape[1]
        for i, number in enumerate(input.shape):
            if i < dims:
                dim_ = index[:, i]
                dim_idx.append(dim_)
                first_dims.append(number)
            else:
                keep_dims.append(number)

        # 为了不影响输入index的最后一维，避免函数副作用
        target_dix = torch.zeros(
            (index.shape[0], ), dtype=torch.long,
            device=input.device) + dim_idx[-1]
        new_shape = (-1, ) + tuple(keep_dims)
        input2 = torch.reshape(input, new_shape)
        mul2 = 1
        for i in range(dims - 1, 0, -1):
            mul2 *= first_dims[i]
            target_dix += mul2 * dim_idx[i - 1]
        o = input2[target_dix]
        return o

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size**2 * 2, :, :]
        mask = offset_mask[:, filter_size**2 * 2:, :, :]
        mask = self.sigmoid(mask)

        # ===================================
        N, in_C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # 1.先对图片x填充得到填充后的图片pad_x
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W),
                            dtype=torch.float32,
                            device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # 卷积核中心点在pad_x中的位置
        rows = torch.arange(
            0, out_W, dtype=torch.float32,
            device=dcn_weight.device) * stride + padding
        cols = torch.arange(
            0, out_H, dtype=torch.float32,
            device=dcn_weight.device) * stride + padding
        rows = rows[None, None, :, None, None].repeat((1, out_H, 1, 1, 1))
        cols = cols[None, :, None, None, None].repeat((1, 1, out_W, 1, 1))
        start_pos_yx = torch.cat(
            [cols, rows],
            dim=-1)  # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = start_pos_yx.repeat(
            (N, 1, 1, kH * kW,
             1))  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :
                                   1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :,
                                   1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kW - 1) // 2
        rows2 = torch.arange(
            0, kW, dtype=torch.float32, device=dcn_weight.device) - half_W
        cols2 = torch.arange(
            0, kH, dtype=torch.float32, device=dcn_weight.device) - half_H
        rows2 = rows2[None, :, None].repeat((kH, 1, 1))
        cols2 = cols2[:, None, None].repeat((1, kW, 1))
        filter_inner_offset_yx = torch.cat([cols2, rows2],
                                           dim=-1)  # [kH, kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = torch.reshape(
            filter_inner_offset_yx,
            (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = filter_inner_offset_yx.repeat(
            (N, out_H, out_W, 1, 1))  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :
                                                       1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :,
                                                       1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

        mask = mask.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*1]
        offset = offset.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*2]
        offset_yx = torch.reshape(
            offset,
            (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终位置。其实也不是最终位置，为了更快速实现DCNv2，还要给y坐标（代表行号）加上图片的偏移来一次性抽取，避免for循环遍历每一张图片。
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)
        ytxt = torch.cat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = pad_x.permute(0, 2, 3, 1)  # [N, pad_x_H, pad_x_W, C]
        pad_x = torch.reshape(
            pad_x, (N * pad_x_H, pad_x_W, in_C))  # [N*pad_x_H, pad_x_W, C]

        ytxt = torch.reshape(
            ytxt, (N * out_H * out_W * kH * kW, 2))  # [N*out_H*out_W*kH*kW, 2]
        _yt = ytxt[:, :1]  # [N*out_H*out_W*kH*kW, 1]
        _xt = ytxt[:, 1:]  # [N*out_H*out_W*kH*kW, 1]

        # 为了避免使用for循环遍历每一张图片，还要给y坐标（代表行号）加上图片的偏移来一次性抽取出更兴趣的像素。
        row_offset = torch.arange(
            0, N, dtype=torch.float32,
            device=dcn_weight.device) * pad_x_H  # [N, ]
        row_offset = row_offset[:, None, None].repeat(
            (1, out_H * out_W * kH * kW, 1))  # [N, out_H*out_W*kH*kW, 1]
        row_offset = torch.reshape(
            row_offset,
            (N * out_H * out_W * kH * kW, 1))  # [N*out_H*out_W*kH*kW, 1]
        row_offset.requires_grad = False
        _yt += row_offset

        _y1 = torch.floor(_yt)
        _x1 = torch.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        _y1x1 = torch.cat([_y1, _x1], -1)
        _y1x2 = torch.cat([_y1, _x2], -1)
        _y2x1 = torch.cat([_y2, _x1], -1)
        _y2x2 = torch.cat([_y2, _x2], -1)

        _y1x1_int = _y1x1.long()  # [N*out_H*out_W*kH*kW, 2]
        v1 = self.gather_nd(pad_x, _y1x1_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y1x2_int = _y1x2.long()  # [N*out_H*out_W*kH*kW, 2]
        v2 = self.gather_nd(pad_x, _y1x2_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y2x1_int = _y2x1.long()  # [N*out_H*out_W*kH*kW, 2]
        v3 = self.gather_nd(pad_x, _y2x1_int)  # [N*out_H*out_W*kH*kW, in_C]
        _y2x2_int = _y2x2.long()  # [N*out_H*out_W*kH*kW, 2]
        v4 = self.gather_nd(pad_x, _y2x2_int)  # [N*out_H*out_W*kH*kW, in_C]

        lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [N*out_H*out_W*kH*kW, in_C]
        mask = torch.reshape(mask, (N * out_H * out_W * kH * kW, 1))
        value = value * mask
        value = torch.reshape(value, (N, out_H, out_W, kH, kW, in_C))
        new_x = value.permute(0, 1, 2, 5, 3,
                              4)  # [N, out_H, out_W, in_C, kH, kW]

        # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
        new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH *
                                      kW))  # [N, out_H, out_W, in_C * kH * kW]
        new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
        rw = torch.reshape(dcn_weight, (
            out_C, in_C * kH * kW, 1,
            1))  # [out_C, in_C, kH, kW] -> [out_C, in_C*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, stride=1)  # [N, out_C, out_H, out_W]
        return out


if __name__ == "__main__":
    model = DCNv2(input_dim=1, filters=1, filter_size=3, stride=1, padding=1)
    nn.init.constant_(model.dcn_weight, 1.0)
    x = torch.arange(25).reshape(1, 1, 5, 5).float()
    out = model(x)
    print(out.squeeze())