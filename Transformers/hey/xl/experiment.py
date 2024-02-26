# -*- coding: utf-8 -*-
# @Time    : 2024/2/25 17:09
# @Author  : GUMP
# @File    : experiment.py
# @Software: PyCharm
from typing import List

import torch
import torch.nn as nn
from labml.logger import Text

from labml import experiment, tracker, monit, logger
from labml.configs import option
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex, hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.xl import TransformerXL, TransformerXLLayer


class AutoregressiveModel(Module):
    def __init__(self, n_vocab: int, d_model: int, transformer: TransformerXL):
        super().__init__()
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask_x = None
        self.mask_mem = None

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor]):
        m_len = len(mem[0]) if mem else 0
        if self.mask_x is None or self.mask_x.shape[0] < len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask_x = subsequent_mask(len(x)).to(x.device)

        if self.mask_mem is None or self.mask_mem.shape[1] < m_len or self.mask_mem.shape[0] < len(x):
            self.mask_mem = self.mask_x.new_ones(len(x), m_len, 1)

        if m_len:
            mask = torch.cat((self.mask_mem[:len(x), :m_len], self.mask_x[:len(x), :len(x)]), dim=1)
        else:
            mask = self.mask_x[:len(x), :len(x)]

        x = self.src_embed(x)
        res, mem = self.transformer(x, mem, mask)
        res = self.generator(res)
        return res, mem


class Configs(NLPAutoRegressionConfigs):
    model: AutoregressiveModel
    d_model: int = 128
    heads: int = 4
    dropout: float = 0.0
    d_ff: int = 256
    n_layers: int = 6
    mem_len: int = 128

    memory = SimpleStateModule()

    def init(self):
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)

        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy, self.memory]

    def merge_memory(self, old_mem, new_mem):
        if self.mem_len == 0:
            return []

        if old_mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(old_mem, new_mem)]
        else:
            mem = new_mem

        if len(mem[0]) > self.mem_len:
            mem = [m[-self.mem_len:] for m in mem]

        return mem

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        with self.mode.update(is_log_activations=batch_idx.is_last):
            mem = self.memory.get()
            output, new_mem = self.model(data, mem)
            mem = self.merge_memory(mem, new_mem)
            self.memory.set(mem)

        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        self.accuracy(output, target)
        self.accuracy.track()

        if self.mode.is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.model)

        tracker.save()

    def sample(self):
        prompt = self.prompt
        log = [(prompt, Text.subtle)]
        mem = []
        for i in monit.iterate('Sample', 25):
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            output, new_mem = self.model(data, mem)
            output = output.argmax(dim=-1).squeeze(1)
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            prompt = prompt[-1:]
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]
            mem = self.merge_memory(mem, new_mem)
        logger.log(log)


@option(Configs.model)
def autoregressive_model(c: Configs):
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    m = AutoregressiveModel(c.n_tokens, c.d_model, TransformerXL(
        TransformerXLLayer(d_model=c.d_model,
                           self_attn=RelativeMultiHeadAttention(c.heads, c.d_model, c.dropout),
                           feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                           dropout_prob=c.dropout), c.n_layers))
    return m.to(c.device)


def main():
    experiment.create(name="transformer_xl", comment='')
    conf = Configs()
    experiment.configs(conf,
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'sequential_train_loader',
                        'valid_loader': 'sequential_valid_loader',

                        'seq_len': 2,
                        'mem_len': 32,
                        'epochs': 128,
                        'batch_size': 32,
                        'inner_iterations': 25})
    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
