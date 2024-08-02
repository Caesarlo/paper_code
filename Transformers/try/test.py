# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 22:00
# @Author  : GUMP
# @File    : test.py
# @Software: PyCharm
import torch
from transformer_ import Transformer
from transformer_ import create_masks
# 完整示例

# 定义一些参数
num_layers = 4
d_model = 128
num_heads = 8
d_ff = 512
input_vocab_size = 8500
target_vocab_size = 8000
max_len = 100
dropout = 0.1

# 创建 Transformer 模型
transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout)

# 创建输入和目标序列
enc_input = torch.tensor([[1, 2, 3, 4, 0, 0]])
dec_input = torch.tensor([[1, 2, 3, 4, 0]])

# 创建掩码
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_input, dec_input)

# 前向传播
outputs = transformer(enc_input, dec_input, enc_padding_mask, combined_mask, dec_padding_mask)
print(outputs)
