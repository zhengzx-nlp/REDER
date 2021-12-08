import sys
from typing import Optional

import torch
import torch.nn as nn
from fairseq.modules import MultiheadAttention
from nonauto.modules.attention import MultiheadRelativeAttention
from nonauto.modules.reversible_utils import *

from .model_utils import *


class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, dropout, fn):
        super().__init__()
        self.norm = norm_class(dim, elementwise_affine=False)
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        if isinstance(x, tuple):
            x = list(x)
            x[0] = self.dropout(x[0])
            return tuple(x)
        return self.dropout(x)


class SandwichNorm(nn.Module):
    def __init__(self, norm_class, dim, dropout, fn):
        super().__init__()
        self.prenorm = norm_class(dim)
        self.postnorm = norm_class(dim)
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.prenorm(x)
        x = self.fn(x, **kwargs)
        if isinstance(x, tuple):
            x = list(x)
            x[0] = self.dropout(self.postnorm(x[0]))
            return tuple(x)
        return self.dropout(self.postnorm(x))


RESIDUAL_FN_WRAPPERS = {
    "prenorm": partial(PreNorm, LayerNorm),
    "sandwich": partial(SandwichNorm, LayerNorm)
}


class FeedForward(nn.Module):
    def __init__(self, dim, dim_inner=4, dropout=0.0, activation=nn.GELU, glu=False):
        super().__init__()
        self.glu = glu
        self.w1 = nn.Linear(dim, dim_inner * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim_inner, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class LinearUpsampler(nn.Module):
    def __init__(self, input_size, s=3) -> None:
        super(LinearUpsampler, self).__init__()

        self._input_size = input_size
        self._s = s
        self.mlp = nn.Linear(input_size, input_size * s)
        nn.init.xavier_normal_(self.mlp.weight)

    def forward(self, x, mask):
        # mask: [B, T]
        (B, T, D) = x.shape

        _x = self.mlp(x)
        _x = _x.reshape(B, T*self._s, D)

        _mask = mask.unsqueeze(-1).expand(B, T, self._s)
        _mask = _mask.reshape(B, -1)

        assert _x.size(1) == _mask.size(1)

        return _x, _mask


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx',
                             nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(
            torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[..., self.forward_shuffle_idx]
        else:
            return x[..., self.backward_shuffle_idx]

    def reverse(self, x):
        return self(x, reverse=True)


class ActNorm(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.log_scale = nn.Parameter(torch.zeros(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))

        self.register_buffer("_initialized", torch.tensor(False))

    def initialize(self, input, mask):
        # input: [bsz, L, d]
        # mask: [bsz, L]
        num = mask.sum()
        with torch.no_grad():
            data = input.clone().mul(mask[..., None])
            mean = torch.sum(data, dim=[0, 1]) / num
            vars = torch.sum((data - mean) ** 2, dim=[0, 1]) / num
            inv_std = 1 / (torch.sqrt(vars) + 1e-6)

            self.bias.data.copy_(-mean.data)
            self.log_scale.data.copy_(inv_std.log().data)

            self._initialized = torch.tensor(True)

    def forward(self, input, mask):
        if not self._initialized:
            self.initialize(input, mask)
        out = input * self.log_scale.exp() + self.bias
        return out

    def reverse(self, output, mask=None):
        return (output - self.bias).div(self.log_scale.exp() + 1e-6)


class RevTransformerEncoderLayer(nn.Module):
    """
    RevNet style Transformer layer
    Inspired by implementation of Reformer 
    "Reformer: The Efficient Transformer" (https://github.com/lucidrains/reformer-pytorch)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim

        self_attn = self.build_self_attention(self.embed_dim, args)
        self_attn.forward = lambda x, **kwargs: self_attn.__class__.forward(
            self_attn, query=x, key=x, value=x, **kwargs
        )

        feed_forward = self.build_ffn(self.embed_dim, args)

        residual_fn_wrapper = RESIDUAL_FN_WRAPPERS[args.layer_norm_type]
        self.self_attn = residual_fn_wrapper(
            self.embed_dim, args.dropout, self_attn)
        self.feed_forward = residual_fn_wrapper(
            self.embed_dim, args.dropout, feed_forward)

        # dropout requires to have the same
        # seed for forward and backward pass
        self.self_attn_seed = None
        self.feed_forward_seed = None

        self.reuse_seed = False

    def build_self_attention(self, embed_dim, args):
        attn_cls = {
            'normal': MultiheadAttention,
            'relative': MultiheadRelativeAttention
        }[args.self_attention_type]

        return attn_cls(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            max_position=self.args.max_source_positions
        )

    def build_ffn(self, embed_dim, args):
        return FeedForward(
            embed_dim, 
            args.encoder_ffn_embed_dim,
            dropout=args.dropout, 
            activation=nn.GELU
        )

    def _init_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(seed)
        return seed

    def f(
        self,
        prev_hidden_states,
        self_attn_padding_mask=None,
        self_attn_mask=None,
        rel_attn_kv=None,
    ):
        # self attention
        if self.reuse_seed:
            torch.manual_seed(self.self_attn_seed)
        elif self.training:
            self.self_attn_seed = self._init_seed()

        self_attn_output, attn_weights = self.self_attn(
            prev_hidden_states,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            rel_attn_kv=rel_attn_kv
        )

        return self_attn_output, attn_weights

    def g(self, attn_output):
        if self.reuse_seed:
            torch.manual_seed(self.feed_forward_seed)
        elif self.training:
            self.feed_forward_seed = self._init_seed()
        return self.feed_forward(attn_output)

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
        rel_attn_kv: list = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # every forward pass we sample a different seed
        # for dropout and save for forward fn in backward pass
        # to have correct dropout

        """ X_1,            X_2 """
        prev_attn_output, prev_hidden_states = torch.chunk(x, chunks=2, dim=-1)

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_1 = X_1 + f(X_2)
        # f(X_2) = self_attn(X_2)
        attn_output, attn_weights = self.f(
            prev_hidden_states=prev_hidden_states,
            self_attn_padding_mask=encoder_padding_mask,
            self_attn_mask=attn_mask,
            rel_attn_kv=rel_attn_kv,
        )
        """ Y_1 = X_1 + f(X_2) """
        attn_output = prev_attn_output + attn_output

        # # free memory
        # del prev_attn_output

        # Y_2 = X_2 + g(Y_1)
        # g(Y_1) = FFN(Y_1)
        ffn_output = self.g(attn_output)
        """ Y_2       =        X_2         +   g(Y_1) """
        hidden_states = prev_hidden_states + ffn_output

        out = torch.cat([attn_output, hidden_states], dim=-1)

        return out

    def reverse(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
        rel_attn_kv: list = None
    ):
        # every forward pass we sample a different seed
        # for dropout and save for forward fn in backward pass
        # to have correct dropout
        """ Y_1,            Y_2 """
        prev_attn_output, prev_hidden_states = torch.chunk(x, chunks=2, dim=-1)

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_2 = X_2 + g(Y_1)
        # g(Y_1) = FFN(Y_1)
        ffn_output = self.g(prev_attn_output)
        """ X_2       =        Y_2         -   g(Y_1) """
        hidden_states = prev_hidden_states - ffn_output

        # Y_1 = X_1 + f(X_2)
        # f(X_2) = self_attn(X_2)
        attn_output, attn_weights = self.f(
            prev_hidden_states=hidden_states,
            self_attn_padding_mask=encoder_padding_mask,
            self_attn_mask=attn_mask,
            rel_attn_kv=rel_attn_kv,
        )
        """ X_1 = Y_1 - f(X_2) """
        attn_output = prev_attn_output - attn_output

        # # free memory
        # del prev_attn_output

        out = torch.cat([attn_output, hidden_states], dim=-1)

        return out

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]


class RevTransformerLangEnd(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_layers = self.args.encoder_layers
        self.layers = nn.ModuleList(
            [RevTransformerEncoderLayer(args) for _ in range(self.num_layers)]
        )

        if self.args.feature_shuffle:
            self.shuffles = nn.ModuleList(
                [Shuffle(in_channels=self.args.encoder_embed_dim*2)
                 for _ in range(self.num_layers)]
            )

    def forward(self, x, padding_mask, **kwargs):
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []

        for i in range(self.num_layers):
            if self.args.feature_shuffle:
                x = self.shuffles[i].forward(x)
            x = self.layers[i].forward(
                x,
                padding_mask,
                **kwargs
            )
            inner_states.append(x.transpose(0, 1))

        return x.transpose(0, 1), inner_states

    def reverse(self, x, padding_mask, **kwargs):
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []

        for i in reversed(range(self.num_layers)):
            x = self.layers[i].reverse(
                x,
                padding_mask,
                **kwargs
            )
            if self.args.feature_shuffle:
                x = self.shuffles[i].reverse(x)
            inner_states.append(x.transpose(0, 1))

        return x.transpose(0, 1), inner_states
