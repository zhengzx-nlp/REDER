import math
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq.modules.multihead_attention import MultiheadAttention
from nonauto.modules.position_embedding import RelativePositionEmbedding
from torch import Tensor, nn


class MultiheadRelativeAttention(MultiheadAttention):
    def __init__(self, max_position=32, **kwargs):
        super().__init__(**kwargs)

        embed_dim, num_heads = kwargs['embed_dim'], kwargs["num_heads"]
        head_dim = embed_dim // num_heads
        self.rel_pos_key = RelativePositionEmbedding(
            max_position, head_dim)
        # self.rel_pos_values = RelativePositionEmbedding(
        #     max_position, head_dim)

        
    def _compute_relative_attention(self, q, k, v, key_padding_mask, attn_mask, rel_k, rel_v):
        """Calculate relative position-aware dot-product self-attention.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v.
        \alpha = softmax( q(k+rel_k) ); out = \alpha (v+rel_v)
        Args:
            q: a Tensor with shape [batch, heads, qlen, depth].
            k: a Tensor with shape [batch, heads, klen, depth].
            v: a Tensor with shape [batch, heads, klen, depth].
            bias: bias Tensor.
            relative_embedding_keys: a Tensor with shape [(bsz), qlen, klen, depth].
            relative_embedding_values: a Tensor with shape [(bsz), qlen, klen, depth].
            dropout (optional): nn.Dropout.
        Returns:
            Attention weights. [batch, heads, qlen, klen]
            Attention outputs. [batch, heads, qlen, depth]
        """
        QK = torch.einsum("bhqd,bhkd->bhqk", [q, k])
        if rel_k.dim() == 3:
            QR = torch.einsum("bhqd,qkd->bhqk", [q, rel_k])
        elif rel_k.dim() == 4:
            QR = torch.einsum("bhqd,bqkd->bhqk", [q, rel_k])
        logits = QK + QR
        logits *= self.scaling

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            logits += attn_mask

        # [bsz, head, qlen, klen]
        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )

        alpha = F.softmax(logits, -1, dtype=torch.float32).type_as(logits)
        alpha = self.dropout_module(alpha)

        AV = torch.einsum("bhqk,bhkd->bhqd", [alpha, v])
        # if rel_v.dim() == 3:
        #     AR = torch.einsum("bhqk,qkd->bhqd", [alpha, rel_v])
        # elif rel_v.dim() == 4:
        #     AR = torch.einsum("bhqk,bqkd->bhqd", [alpha, rel_v])
        # out = AV + AR
        out = AV

        return out, alpha

    def _split_heads(self, x):
        # [len, bsz, heads*d]
        batch_size, seq_len = x.size(1), x.size(0)

        # [bsz, heads, len, d]
        return (
            x.view(seq_len, batch_size, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

    def _combine_heads(self, x):
        """:param x: [batch_size, head_count, seq_len, dim_per_head]"""
        batch_size, seq_len = x.size(0), x.size(2)

        # [len, bsz, heads*d]
        return (
            x.permute(2, 0, 1, 3)
            .contiguous()
            .view(seq_len, batch_size, self.num_heads * self.head_dim)
        )

    def _prepare_relative(self, emb):
        # relative attention related
        length = emb.size(0)
        # return [self.rel_pos_key(length), self.rel_pos_values(length)]
        return [self.rel_pos_key(length), None]

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rel_attn_kv: list = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if rel_attn_kv is None:
            rel_attn_kv = self._prepare_relative(query)

        # 1) Project key, value, and query.
        # [batch_size, num_head, seq_len, dim_head]
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # 2) Calculate and scale scores.
        # attn ([batch, num_head, length, d_model])
        attn, attn_weights = self._compute_relative_attention(
            q=q, k=k, v=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rel_k=rel_attn_kv[0],
            rel_v=rel_attn_kv[1],
        )

        # 3) Apply attention dropout and compute context vectors.
        # attn ([batch, length, d_model])
        attn = self._combine_heads(attn)
        attn = self.out_proj(attn)

        return attn, attn_weights
