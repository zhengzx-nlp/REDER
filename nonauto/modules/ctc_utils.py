from fairseq.data.data_utils import collate_tokens
import torch
from torch._C import device
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.nn import functional as F
import sys


def convert_alignment_to_symbol(aligns, targets, blank, padding_mask, pad):
    def get_symbol(align, target):
        def _gs(a):
            if a % 2 == 0:
                symbol = blank
            else:
                symbol = target[a // 2]
            return symbol
        return torch.tensor(list(map(_gs, align)))
    symbols = collate_tokens(list(map(get_symbol, aligns, targets)),
                             pad, left_pad=False).clone().to(padding_mask.device)
    return symbols


def post_process_ctc_decoded_tokens_with_pad(ctc_decoded_batch_tokens, pad, left_pad=False):
    pp_decoded_tokens = post_process_ctc_decoded_tokens(ctc_decoded_batch_tokens)
    padded_batch_tokens = collate_tokens(pp_decoded_tokens, pad, left_pad=left_pad)
    return padded_batch_tokens


def post_process_ctc_decoded_tokens(ctc_decoded_batch_tokens):
    # ctc_decoded_tokens: (bsz, len)

    def merge(_toks):
        return _toks.new_tensor(
            [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i-1]]
        )

    return [merge(tokens) for tokens in ctc_decoded_batch_tokens]
