# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 22:04
# @Author  : GUMP
# @File    : dialogue.py
# @Software: PyCharm
import torch
from transformer_ import create_masks


def generate_response(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    dec_input = torch.tensor([[tokenizer.bos_token_id]])

    for _ in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_ids, dec_input)
        predictions = model(input_ids, dec_input, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :].squeeze(1)
        predicted_id = torch.argmax(predictions, dim=-1).item()

        if predicted_id == tokenizer.eos_token_id:
            break

        dec_input = torch.cat([dec_input, torch.tensor([[predicted_id]])], dim=-1)

    response = tokenizer.decode(dec_input.squeeze().tolist(), skip_special_tokens=True)
    return response


input_text = "Hello, how are you?"
response = generate_response(model, tokenizer, input_text)
print(response)
