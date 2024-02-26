import copy

import torch.nn as nn

from labml.configs import BaseConfigs, option, calculate, aggregate
from labml_helpers.module import Module
from feed_forward import FeedForward
from mha import MultiHeadAttention
from models import EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, TransformerLayer, \
    Encoder, Decoder, Generator, EncoderDecoder


class FeedForwardConfigs(BaseConfigs):
    ffn: FeedForward
    d_model: int
    d_ff: int = 2048
    dropout: float = 0.1
    activation: nn.Module = 'ReLU'
    is_gated: bool = False
    bias1: bool = True
    bias2: bool = True
    bias_gate: bool = False
    glu_variant: str = 'none'


@option(FeedForwardConfigs.activation, 'ReLU')
def _ffn_activation_relu():
    return nn.ReLU()


@option(FeedForwardConfigs.activation, 'GELU')
def _ffn_activation_gelu():
    return nn.DELU()


@option(FeedForwardConfigs.ffn, 'default')
def _feed_forward(c: FeedForwardConfigs):
    return FeedForward(c.d_model, c.d_ff,
                       dropout=c.dropout,
                       activation=c.activation,
                       is_gated=c.is_gated,
                       bias1=c.bias1,
                       bias2=c.bias2,
                       bias_gate=c.bias_gate)


# FFN with Bilinear hidden layer
aggregate(FeedForwardConfigs.glu_variant, 'GLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.Sigmoid()))

# FFN with ReLU gate
aggregate(FeedForwardConfigs.glu_variant, 'ReGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.ReLU()))

# FFN with GELU gate
aggregate(FeedForwardConfigs.glu_variant, 'GEGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.GELU()))

# FFN with Swish gate
aggregate(FeedForwardConfigs.glu_variant, 'SwiGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.SiLU()))


class TransformerConfigs(BaseConfigs):
    n_heads: int = 8
    d_model: int = 512
    n_layers: int = 6
    dropout: float = 0.1
    n_src_vocab: int
    n_tgt_vocab: int

    encoder_attn: MultiHeadAttention = 'mha'
    decoder_attn: MultiHeadAttention = 'mha'
    decoder_mem_attn: MultiHeadAttention = 'mha'

    ffn: FeedForwardConfigs

    encoder_layer: TransformerLayer = 'default'
    decoder_layer: TransformerLayer = 'default'

    encoder: Encoder = 'default'
    decoder: Decoder = 'default'

    src_embed: Module = 'fixed_pos'
    tgt_embed: Module = 'fixed_pos'

    generator: Generator = 'default'

    encoder_decoder: EncoderDecoder


# Multi-head Attention
def _mha(c: TransformerConfigs):
    return MultiHeadAttention(c.n_heads, c.d_model, dropout_pob=c.dopout)


calculate(TransformerConfigs.encoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_mem_attn, 'mha', _mha)


# Relative Multi-head Attention
def _relative_mha(c: TransformerConfigs):
    from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
    return RelativeMultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'relative', _relative_mha)


@option(TransformerConfigs.ffn, 'default')
def _feed_forward(c: TransformerConfigs):
    conf = FeedForwardConfigs()
    conf.set_default(FeedForwardConfigs.d_model, func=lambda: c.d_model)
    conf.set_default(FeedForwardConfigs.dropout, func=lambda: c.dropout)
    return conf


@option(TransformerConfigs.decoder_layer, 'default')
def _decoder_layer(c: TransformerConfigs):
    return TransformerLayer(d_model=c.d_model, self_attn=c.decoder_attn,
                            src_attn=c.decoder_mem_attn, feed_forward=copy.deepcopy(c.ffn.ffn),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.encoder, 'default')
def _encoder(c: TransformerConfigs):
    return Encoder(c.encoder_layer, c.n_layers)


@option(TransformerConfigs.decoder, 'default')
def _decoder(c: TransformerConfigs):
    return Decoder(c.decoder_layer, c.n_layers)


@option(TransformerConfigs.generator, 'default')
def _generator(c: TransformerConfigs):
    return Generator(c.n_tgt_vocab, c.d_model)


# Fixed Positional Embeddings
@option(TransformerConfigs.src_embed, 'fixed_pos')
def _src_embed_with_positional(c: TransformerConfigs):
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'fixed_pos')
def _tgt_embed_with_positional(c: TransformerConfigs):
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_tgt_vocab)


# Learned Positional Embeddings
@option(TransformerConfigs.src_embed, 'learned_pos')
def _src_embed_with_learned_positional(c: TransformerConfigs):
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'learned_pos')
def _tgt_embed_with_learned_positional(c: TransformerConfigs):
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_tgt_vocab)


# No Positional Embeddings
@option(TransformerConfigs.src_embed, 'no_pos')
def _src_embed_without_positional(c: TransformerConfigs):
    return nn.Embedding(c.n_src_vocab, c.d_model)


@option(TransformerConfigs.tgt_embed, 'no_pos')
def _tgt_embed_without_positional(c: TransformerConfigs):
    return nn.Embedding(c.n_tgt_vocab, c.d_model)


@option(TransformerConfigs.encoder_decoder, 'default')
def _encoder_decoder(c: TransformerConfigs):
    return EncoderDecoder(c.encoder, c.decoder, c.src_embed, c.tgt_embed, c.generator)
