import torch


def subsequent_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask


def _subsequent_mask():
    from labml.logger import inspect
    inspect(subsequent_mask(10)[:, :, 0])


if __name__ == "__main__":
    _subsequent_mask()
