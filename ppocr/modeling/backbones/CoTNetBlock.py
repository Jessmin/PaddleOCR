# -*- coding: utf-8 -*-

# @Time : 2021/8/3 11:35 下午

# @Author : Tombili
# @Email : monlilirua@gmail.com
# @File : CoTNetBlock.py
# @Address : 河南大学 计算机与信息工程学院
# 版权没有，随便侵权

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F


paddle.set_device('cpu')
class CoTNetLayer(nn.Layer):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取上下文信息，视作输入X的静态上下文表达
            nn.Conv2D(dim,
                      dim,
                      kernel_size=kernel_size,
                      padding=1,
                      stride=1,
                      bias_attr=False),
            nn.BatchNorm2D(dim),
            nn.ReLU())
        self.value_embed = nn.Sequential(
            nn.Conv2D(dim, dim, kernel_size=1, stride=1,
                      bias_attr=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2D(dim))

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2D(2 * dim, 2 * dim // factor, 1,
                      bias_attr=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2D(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2D(2 * dim // factor,
                      kernel_size * kernel_size * dim,
                      1,
                      stride=1)  # out: H * W * (K*K*C)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        v = paddle.fluid.layers.reshape(self.value_embed(x), (bs, c, -1))
        # v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码
        y = paddle.fluid.layers.concat(
            [k1, x], axis=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = paddle.fluid.layers.reshape(att, (bs, c, self.kernel_size * self.kernel_size, h, w))
        att = paddle.fluid.layers.reshape(paddle.fluid.layers.reduce_mean(att,2,keep_dim=False), (bs, c, -1))
        # att = att.mean(2, keepdim=False).view(bs, c,-1)  # shape：bs,c,h*w  求平均降低维度
        print(att.shape)
        print(v.shape)
        k2 = F.softmax(att, axis=-1) * v  # 对每一个H*w进行softmax后
        k2 = paddle.fluid.layers.reshape(k2, (bs, c, h, w))
        # k2 = k2.view(bs, c, h, w)

        return k1 + k2  # 注意力融合


if __name__ == '__main__':
    input = paddle.zeros([50, 512, 7, 7])
    cot = CoTNetLayer(dim=512, kernel_size=3)
    output = cot(input)
    print(output.shape)