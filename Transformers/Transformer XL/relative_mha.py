# -*- coding: utf-8 -*-
# @Time    : 2024/1/29 21:11
# @Author  : GUMP
# @File    : relative_mha.py
# @Software: PyCharm

"""
Relative Multi-Headed Attention
This is an implementation of relative multi-headed attention from paper Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context in PyTorch
"""
import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention

"""
This method shifts i^{th} row of a matrix by i columns.
If the input is [[1, 2 ,3], [4, 5 ,6], [7, 8, 9]] , the shifted result would be [[1, 2 ,3], [0, 4, 5], [6, 0, 7]] . Ideally we should mask out the lower triangle but it's ok for our purpose.
"""


def shift_right(x: torch.Tensor):
    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    return x
