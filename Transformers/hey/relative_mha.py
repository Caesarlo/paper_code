import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


def shift_right(x: torch.Tensor):
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob, bias=False)
        self.P = 2 ** 12
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)
       
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = shift_right(b + d)
        bd = bd[:, -key.shape[0]:]

        return ac + bd


def _test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inspect(x)
    inspect(shift_right(x))

    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])


if __name__ == "__main__":
    _test_shift_right()