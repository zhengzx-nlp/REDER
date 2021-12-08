from inspect import stack
from nonauto.modules.position_embedding import RelativePositionEmbedding
from nonauto.modules.attention import MultiHeadedAttentionRelative
import sys
from functools import partial, reduce, wraps
from typing import Any, Dict, List, Optional

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    Embedding,
    Linear,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerModel,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_iwslt_de_en_small,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from nonauto.modules.reversible_utils import ActNorm, InvertibleLinear
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def mse_loss(output, target, mask):
    loss = F.mse_loss(output, target, reduce=False)
    loss.masked_fill_(~mask, 0.0)
    loss = loss.sum()
    return loss


def cos_distance_loss(output, target, mask):
    loss = 1 - F.cosine_similarity(output, target, dim=-1)
    loss.masked_fill_(~mask, 0.0)
    loss = loss.mean(1).mean()
    return loss


class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, dropout, fn):
        super().__init__()
        self.norm = norm_class(dim)
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


residual_fn_wrapper = partial(PreNorm, LayerNorm)


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


@register_model("rev_transformer")
class RevTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

        self.apply_fba_loss = False
        self._word_pred_valid_loss = float("inf")

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--disable-reversible-encoder",
            action="store_true",
            default=False,
            help="use normal transformer encoder",
        )
        parser.add_argument(
            "--disable-reversible-decoder",
            action="store_true",
            default=False,
            help="use normal transformer decoder",
        )
        parser.add_argument(
            "--decoder-postprocess", type=str, default="", help="how to postprocess decoder output"
        )
        parser.add_argument(
            "--forward-backward-agreement",
            action="store_true",
            default=False,
            help="use normal transformer decoder",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if args.disable_reversible_encoder:
            return TransformerEncoder(args, src_dict, embed_tokens)
        else:
            return RevTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.disable_reversible_decoder:
            return TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),
            )
        else:
            return RevTransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),
            )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_tokens=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        net_output = {
            "word_pred": {
                "out": decoder_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
        }
        if self.args.forward_backward_agreement:
            self.apply_fba_loss = self.apply_fba_loss or self._word_pred_valid_loss < 3.0

            if self.apply_fba_loss:
                #                 with torch.no_grad():
                rev_decoder_out, rev_decoder_extra = self.decoder.reverse(
                    tgt_tokens,
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )
                fwd_hiddens = extra["inner_states"]
                bwd_hiddens = rev_decoder_extra["inner_states"]
                fba_loss = cos_distance_loss(
                    output=torch.stack(fwd_hiddens, 2).transpose(1, 0),
                    target=torch.stack(bwd_hiddens, 2).transpose(1, 0).detach(),
                    mask=tgt_tokens.ne(self.pad)[:, :, None, None],
                )
            else:
                fba_loss = torch.tensor(0.0).to(decoder_out)

            net_output["fba_cos_loss"] = {"loss": fba_loss, "factor": 0.1}

        return net_output


class RevTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args
        self.embed_dim = embed_tokens.embedding_dim
        # if self.layer_norm is not None:
        #     self.layer_norm = LayerNorm(self.embed_dim*2)
        self.project_model_dim = Linear(self.embed_dim * 2, self.embed_dim)

        if args.self_attention_type == 'relative':
            head_dim = self.args.encoder_embed_dim // self.args.encoder_attention_heads
            self.rel_pos_key = RelativePositionEmbedding(args.max_source_positions, head_dim)
            self.rel_pos_values = RelativePositionEmbedding(args.max_source_positions, head_dim)

    def _prepare_relative(self, length):
        # relative attention related
        return self.rel_pos_key(length), self.rel_pos_values(length)

    def build_encoder_layer(self, args):
        return RevTransformerEncoderLayer(args)

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        if self.args.self_attention_type == 'relative':
            src_len = src_tokens.size(1)
            rel_attn_k, rel_attn_v = self._prepare_relative(src_len)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        # concat same tensor for revnet.
        x = torch.cat([x, x], dim=-1)
        for layer in self.layers:
            # x = layer(x, encoder_padding_mask)
            if self.args.self_attention_type == 'relative':
                x = layer(x, encoder_padding_mask, rel_attn_kv=[rel_attn_k, rel_attn_v])
            else:
                x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        # map the doubled size back to d_embded
        x = self.project_model_dim(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class RevTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim

        self_attn = self.build_self_attention(self.embed_dim, args)
        self_attn.forward = lambda x, **kwargs: self_attn.__class__.forward(
            self_attn, query=x, key=x, value=x, **kwargs
        )

        feed_forward = self.build_ffn(self.embed_dim, args)

        self.self_attn = residual_fn_wrapper(self.embed_dim, args.dropout, self_attn)
        self.feed_forward = residual_fn_wrapper(self.embed_dim, args.dropout, feed_forward)

        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

    def build_self_attention(self, embed_dim, args):
        attn_cls = {
            'normal': MultiheadAttention, 
            'relative': MultiHeadedAttentionRelative
        }[args.self_attention_type]

        return attn_cls(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def build_ffn(self, embed_dim, args):
        return FeedForward(
            embed_dim, args.encoder_ffn_embed_dim, dropout=args.dropout, activation=nn.GELU
        )

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.feed_forward_seed)

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, rel_attn_kv: list = None):
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
        if self.training:
            self._init_attention_seed()

        # X_1,            X_2
        prev_attn_output, prev_hidden_states = torch.chunk(x, chunks=2, dim=-1)

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_1 = X_1 + f(X_2)
        # f(X_2)
        if self.args.self_attention_type == 'normal':
            attn_output, attn_weights = self.self_attn(
                prev_hidden_states, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask,
            )
        elif self.args.self_attention_type == 'relative': 
            attn_output, attn_weights = self.self_attn(
                prev_hidden_states, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask, rel_attn_kv=rel_attn_kv
            )
        else:
            raise Exception(f"unexpected self attention type: {self.args.self_attention_type}")
        """ Y_1 = X_1 + f(X_2) """
        attn_output = prev_attn_output + attn_output

        # # free memory
        # del prev_attn_output

        # every forward pass we sample a different seed
        # for dropout and save seed for forward fn in backward
        # to have correct dropout
        if self.training:
            self._init_feed_forward_seed()
        """ Y_2 = X_2 + g(Y_1) """
        hidden_states = prev_hidden_states + self.feed_forward(attn_output)

        out = torch.cat([attn_output, hidden_states], dim=-1)

        return out

    def reverse(
        self, out, encoder_padding_mask, attn_mask: Optional[Tensor] = None, requires_grad=False
    ):
        # Implements the backward pass for reversible ResNets.
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

        with torch.set_grad_enabled(requires_grad):
            """ Y_1             Y_2 """
            next_attn_output, next_hidden_states = torch.chunk(out, chunks=2, dim=-1)

            # set seed to have correct dropout
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1)
            res_hidden_states = self.feed_forward(next_attn_output)
            # res_hidden_states.backward(grad_hidden_states, retain_graph=True)

            """ NOTE X_2 = Y_2 - g(Y_1) """
            hidden_states = next_hidden_states - res_hidden_states
            del res_hidden_states

            # grad_attn_output = grad_attn_output + next_attn_output.grad
            # next_attn_output.grad = None

            # hidden_states.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.attention_seed)
            # f(X_2)
            attn_output, _ = self.self_attn(
                hidden_states,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

            """ X_1 = Y_1 - f(X_2) """
            attn_output = next_attn_output - attn_output
            del next_attn_output

            x = torch.cat([attn_output, hidden_states], dim=-1)
        return x


class RevTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        # self.invertible_project_model_dim = InvertibleLinear(self.embed_dim*2, self.embed_dim)
        if args.decoder_postprocess == "average_actnorm":
            self.inv_output_project = InvertibleLinear(self.embed_dim * 2)
            self.actnorm = ActNorm(self.embed_dim * 2)
            self.U_inv = Linear(self.embed_dim * 2, self.embed_dim * 2, False)
        elif args.decoder_postprocess == "only_hidden":
            pass
        else:
            self.project_model_dim = Linear(self.embed_dim * 2, self.embed_dim)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return RevTransformerDecoderLayer(args, no_encoder_attn)

    def postprocess_layer(self, x, reverse=False):
        if self.args.decoder_postprocess == "average_actnorm":
            if not reverse:
                # [o_1, o_2] = f(x) = f([attn, hidden])
                # [o_1, o_2] @ [e, e].T = (o_1+o_2) @ e.T
                o = self.inv_output_project(x)
                o = self.actnorm(o)
                o_1, o_2 = torch.chunk(o, 2, -1)
                return o_1 + o_2
                # return o
            else:
                # x = [embded, embded]
                # -> f^-1([e, e]) => [attn, hidden]
                o = self.actnorm.reverse(x)
                o = self.inv_output_project.reverse(o)
                return o
        elif self.args.decoder_postprocess == "only_hidden":
            attn, hidden_states = torch.chunk(x, 2, -1)
            return self.layer_norm(hidden_states)
        else:
            return self.layer_norm(self.project_model_dim(x))

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens, incremental_state=incremental_state)
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        # concat same tensor for revnet.
        x = torch.cat([x, x], dim=-1)
        init_x = x

        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        x = self.postprocess_layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn], "inner_states": inner_states}

    def reverse(
        self,
        output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        if incremental_state is not None:
            output_tokens = output_tokens[:, -1:]

        # embed tokens and positions
        y_emb = self.embed_tokens(output_tokens).detach()

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = output_tokens.eq(self.padding_idx)

        y = self.U_inv(torch.cat([y_emb, y_emb], -1))
        #         y = torch.cat([y_emb, y_emb], -1)
        with torch.no_grad():
            x, extra = self.reverse_extract_features(
                y,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                self_attn_padding_mask=self_attn_padding_mask,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
            )

        return x, extra

    def reverse_extract_features(
        self,
        output_states,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_padding_mask=None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # B x T x C -> T x B x C
        y = output_states.transpose(0, 1)

        # concat same tensor for revnet.
        y = self.postprocess_layer(y, reverse=True)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [y]
        #         encoder_out_detach = encoder_out.encoder_out.detach()
        for rev_idx, layer in enumerate(self.layers[::-1]):
            idx = len(self.layers) - rev_idx - 1
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(y)
            else:
                self_attn_mask = None

            y, layer_attn, _ = layer.reverse(
                y,
                encoder_out.encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(y)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(y)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        # T x B x C -> B x T x C
        y = y.transpose(0, 1)

        return y, {"attn": [attn], "inner_states": inner_states}


class RevTransformerDecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim

        self_attn = self.build_self_attention(self.embed_dim, args)
        self_attn.forward = lambda x, **kwargs: MultiheadAttention.forward(
            self_attn, query=x, key=x, value=x, **kwargs
        )

        encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        encoder_attn.forward = lambda x, **kwargs: MultiheadAttention.forward(
            encoder_attn, query=x, **kwargs
        )

        feed_forward = self.build_ffn(self.embed_dim, args)

        self.self_attn = residual_fn_wrapper(self.embed_dim, args.dropout, self_attn)
        self.encoder_attn = residual_fn_wrapper(self.embed_dim, args.dropout, encoder_attn)
        self.feed_forward = residual_fn_wrapper(self.embed_dim, args.dropout, feed_forward)

        # dropout requires to have the same
        # seed for forward and backward pass
        self.self_attn_seed = None
        self.encoder_attn_seed = None
        self.feed_forward_seed = None

        self.need_attn = True
        self.onnx_trace = False

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def build_ffn(self, embed_dim, args):
        return FeedForward(
            embed_dim, args.encoder_ffn_embed_dim, dropout=args.dropout, activation=nn.GELU
        )

    def _init_seed(self, seed_name):
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
        assert hasattr(self, seed_name)
        setattr(self, seed_name, seed)
        torch.manual_seed(seed)

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def f(
        self,
        prev_hidden_states,
        self_attn_padding_mask=None,
        incremental_state=None,
        self_attn_mask=None,
        encoder_out=None,
        encoder_padding_mask=None,
        need_attn=None,
        need_head_weights=None,
        reverse=False,
    ):
        # self attention
        if reverse:
            torch.manual_seed(self.self_attn_seed)
        elif self.training:
            self._init_seed("self_attn_seed")
        self_attn_output, attn = self.self_attn(
            prev_hidden_states,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        self_attn_output = self_attn_output + prev_hidden_states

        # encoder attention
        if reverse:
            torch.manual_seed(self.encoder_attn_seed)
        elif self.training:
            self._init_seed("encoder_attn_seed")
        encoder_attn_output, attn = self.encoder_attn(
            self_attn_output,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        return encoder_attn_output, attn

    def g(self, attn_output, reverse=False):
        if reverse:
            torch.manual_seed(self.feed_forward_seed)
        elif self.training:
            self._init_seed("feed_forward_seed")
        return self.feed_forward(attn_output)

    def prepare_for_generation(
        self,
        prev_self_attn_state,
        prev_attn_state,
        incremental_state=None,
    ):
        # for self attention
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        # for encoder attention
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            assert incremental_state is not None
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        self.prepare_for_generation(prev_self_attn_state, prev_attn_state, incremental_state)

        """ X_1,            X_2 """
        prev_attn_output, prev_hidden_states = torch.chunk(x, chunks=2, dim=-1)

        #  Y_1 = X_1 + f(X_2)
        #  f(X_2) = encoder_attn(self_attn(X_2))
        encoder_attn_output, attn = self.f(
            prev_hidden_states,
            self_attn_padding_mask,
            incremental_state,
            self_attn_mask,
            encoder_out,
            encoder_padding_mask,
            need_attn,
            need_head_weights,
        )
        """ Y_1     =       X_1        +        f(X_2) """
        attn_output = prev_attn_output + encoder_attn_output

        # Y_2 = X_2 + g(Y_1)
        # g(Y_1) = FFN(Y_1)
        ffn_output = self.g(attn_output)
        """ Y_2       =        X_2         +   g(Y_1) """
        hidden_states = prev_hidden_states + ffn_output

        out = torch.cat([attn_output, hidden_states], dim=-1)

        return out, attn, None

    def reverse(
        self,
        out,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        requires_grad=False,
    ):
        # Implements the backward pass for reversible ResNets.
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
        if need_head_weights:
            need_attn = True

        with torch.set_grad_enabled(requires_grad):
            self.prepare_for_generation(prev_self_attn_state, prev_attn_state, incremental_state)

            """ Y_1,            Y_2 """
            next_attn_output, next_hidden_states = torch.chunk(out, chunks=2, dim=-1)

            # g(Y_1) = ffn
            ffn_output = self.g(next_attn_output, reverse=True)
            """ X_2 = Y_2 - g(Y_1) """
            hidden_states = next_hidden_states - ffn_output

            # f(X_2) = encoder_attn(self_attn(X_2))
            encoder_attn_output, attn = self.f(
                hidden_states,
                self_attn_padding_mask,
                incremental_state,
                self_attn_mask,
                encoder_out,
                encoder_padding_mask,
                need_attn,
                need_head_weights,
                reverse=True,
            )
            """ X_1 = Y_1 - f(X_2) """
            attn_output = next_attn_output - encoder_attn_output

            del ffn_output, next_attn_output

            x = torch.cat([attn_output, hidden_states], dim=-1)
        return x, attn, None


@register_model_architecture("rev_transformer", "rev_transformer_iwslt_de_en")
def rev_transformer_iwslt_de_en(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    transformer_iwslt_de_en(args)


@register_model_architecture("rev_transformer", "rev_transformer_iwslt_de_en_small")
def rev_transformer_iwslt_de_en_small(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    transformer_iwslt_de_en_small(args)
