# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 22:03
# @Author  : GUMP
# @File    : train.py
# @Software: PyCharm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformer_ import Transformer
from data_utils import DialogueDataset, dialogues


def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq[:, None, None, :]


def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones((size, size))).unsqueeze(0).unsqueeze(0)
    return mask


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


# 创建数据集和数据加载器
tokenizer = ...  # 使用适当的 tokenizer，例如 GPT-2 tokenizer
dataset = DialogueDataset(dialogues, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
model = Transformer(num_layers=4, d_model=128, num_heads=8, d_ff=512, input_vocab_size=8500, target_vocab_size=8000,
                    max_len=100, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
for epoch in range(10):  # 假设训练10个epoch
    for batch in dataloader:
        inp = batch['input_ids']
        tar = batch['labels']

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        optimizer.zero_grad()
        outputs = model(inp, tar, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), tar.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
