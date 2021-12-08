import math
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiInputPositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1, inp_sizes=[]):
        super().__init__()
        self.total_input_size = size + sum(inp_sizes)
        self.w_1 = nn.Linear(self.total_input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(self.total_input_size)

        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, *x):
        inps = torch.cat(x, -1)
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(inps))))
        output = self.dropout_2(self.w_2(inter))
        return output + x[0]


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, simple_ret=False):

        super(MultiHeadedAttention, self).__init__()

        if dim_per_head is None:
            assert model_dim % head_count == 0
            dim_per_head = model_dim // head_count

        self.head_count = head_count

        self.dim_per_head = dim_per_head

        self.model_dim = model_dim

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(
            self.dim_per_head * head_count, model_dim)

        self.simple_ret = simple_ret

    def _split_heads(self, x):

        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2).contiguous()

    def _combine_heads(self, x):
        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None, enc_attn_cache=None, self_attn_cache=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            # [batch_size, num_head, seq_len, dim_head]
            key_up = self._split_heads(self.linear_keys(key))
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()
        # END CHECK
        if self.simple_ret:
            return output
        return output, top_attn, [key_up, value_up]


class MultiInputGates(nn.Module):
    def __init__(self, d_model, input_sizes=[], dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.total_size = d_model + sum(input_sizes)
        self.num_inputs = 1 + len(input_sizes)

        self.inp_ln = nn.LayerNorm(self.total_size)

        self.gates = nn.Sequential(
            nn.Linear(self.total_size, d_model * self.num_inputs),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        self.linears = nn.Sequential(
            nn.Linear(self.total_size, d_model * self.num_inputs),
        )

        self.dropout = nn.Dropout(dropout)
        # self.out_lns = nn.ModuleList([nn.LayerNorm(d_model)
        #                               for _ in range()])

    def forward(self, *inputs):
        batch, length, _ = inputs[0].size()
        dense = list(inputs)
        inp = self.inp_ln(torch.cat(dense, dim=-1))

        gates = self.gates(inp)
        mid = self.linears(inp)

        out_cat = gates * mid

        out = out_cat.view(batch, length, self.num_inputs, -1).sum(-2)

        # gates = torch.split(gates, self.d_model, dim=-1)
        # # out = sum(map(lambda g, z: g * z, gates, inputs))
        # out = sum([g * ln(z) for g, z, ln in zip(gates, inputs, self.out_lns)])
        # print(len(inputs), len(gates))

        return self.dropout(out) + inputs[0]
