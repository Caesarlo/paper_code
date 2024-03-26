# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 22:49
# @Author  : GUMP
# @File    : experiment.py
# @Software: PyCharm
from labml import experiment
from labml.configs import option, calculate
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import AutoregressiveTransformer, Configs


def _rotary_pe_mha(c: TransformerConfigs):
    from labml_nn.transformers.rope import RotaryPEMultiHeadAttention
    return RotaryPEMultiHeadAttention(c.n_heads, c.d_model, 1.)


calculate(TransformerConfigs.encoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'rotary', _rotary_pe_mha)


@option(Configs.model, 'rotary_pe_transformer')
def _model(c: Configs):
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)
    return m


def main():
    experiment.create(name='rotary_pe_transformer', writers={'screen'})
    conf = Configs()
    experiment.configs(conf, {
        'transformer.src_embed': 'no_pos',
        'transformer.tgt_embed': 'no_pos',
        'transformer.encoder_attn': 'rotary',
        'model': 'rotary_pe_transformer',
        'tokenizer': 'character',
        'prompt_separator': '',
        'prompt': 'It is',
        'text': 'tiny_shakespeare',
        'seq_len': 512,
        'epochs': 32,
        'batch_size': 4,
        'inner_iterations': 10,

    })