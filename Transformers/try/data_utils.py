# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 22:01
# @Author  : GUMP
# @File    : data_utils.py
# @Software: PyCharm


import json
import torch
from torch.utils.data import Dataset


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


data = load_data('path_to_your_dataset.json')

# 假设数据集格式为 [{"input": "Hi!", "response": "Hello!"}, ...]
dialogues = [(item['input'], item['response']) for item in data]


class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        input_text, response_text = self.dialogues[idx]
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').squeeze()
        response_ids = self.tokenizer.encode(response_text, return_tensors='pt').squeeze()
        input_ids = input_ids[:self.max_length // 2]
        response_ids = response_ids[:self.max_length // 2]
        labels = torch.cat([input_ids, response_ids])

        return {'input_ids': labels, 'labels': labels}
