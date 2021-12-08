from collections import namedtuple
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm
from fairseq.utils import new_arange
from nonauto.modules.reversible_utils import *
from torch import Tensor


def apply_bert_random_mask(
    target_tokens,
    vocab,
    mask_prob=0.15,
    leave_unmasked_prob=0.1,
    random_replace_prob=0.1,
):

    pad = vocab.pad()
    bos = vocab.bos()
    eos = vocab.eos()
    unk = vocab.unk()

    target_masks = target_tokens.ne(pad) & \
        target_tokens.ne(bos) & \
        target_tokens.ne(eos)

    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()

    target_length = target_length * mask_prob

    # make sure to mask at least one token.
    target_length = target_length + 1

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(
        target_rank) < target_length[:, None].long()

    mask = target_cutoff.scatter(1, target_rank, target_cutoff)

    sz = mask.size()
    rand_or_unmask_prob = random_replace_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (torch.rand(
            sz, device=mask.device).float() < rand_or_unmask_prob)
        if random_replace_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = torch.rand(sz, device=mask.device) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)

    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    prev_target_tokens = target_tokens.clone()
    prev_target_tokens[mask] = unk

    if rand_mask is not None:
        prev_target_tokens[rand_mask] = \
            target_tokens.clone().float() \
            .random_(vocab.nspecials, len(vocab))

    return prev_target_tokens


DecoderOut = namedtuple(
    'IterativeRefinementDecoderOut',
    [
        'src_tokens',
        'output_tokens',
        'output_scores',
        'attn',
        'step',
        'max_step',
        'history'
    ]
)

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_mid", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)


def _get_lang(reverse):
    return 'tgt' if reverse else 'src'


def tensor_mean_by_mask(t, m):
    # t: bsz, l, d
    # m: bsz, l

    # mean: bsz, d

    m = m.to(t)
    return t.mul(m[..., None]).sum(1) / m.sum(1, keepdim=True)


def convert_mask_to_lengths(mask):
    return mask.long().sum(-1)


def convert_lengths_to_mask(lens, maxlen=None):
    # lens: (bsz)
    maxlen = maxlen or lens.max()
    lens = lens.view(-1)
    mask = torch.arange(maxlen, device=lens.device)[None, :] < lens[:, None]
    return mask

def get_batch_lengths(tokens, pad):
    mask = (tokens != pad)
    return convert_mask_to_lengths(mask)


def cos_distance_loss(output, target, mask):
    B, L = mask.size()
    loss = (0.5 - 0.5 * F.cosine_similarity(output.float(), target.float(), dim=-1)).view(B, L, -1).mean(-1)
    # [B, T]
    loss.masked_fill_(~mask, 0.0)
    loss = (loss.sum(1) / mask.sum(1)).mean()
    return loss


def mse_loss(output, target, mask):
    B, L = mask.size()
    loss = F.mse_loss(output.float(), target.float(), reduction='none').view(B, L, -1).mean(-1)
    # [B, T]
    loss.masked_fill_(~mask, 0.0)
    loss = (loss.sum(1) / mask.sum(1)).mean() 
    return loss


def upgrade_state_dict_with_pretrained_weights(
    state_dict: Dict[str, Any], pretrained_checkpoint: str
) -> Dict[str, Any]:
    import os

    from fairseq import checkpoint_utils

    if not os.path.exists(pretrained_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_checkpoint)
    xlm_state_dict = state["model"]
    for key in state_dict.keys():
        if key in xlm_state_dict:
            print("Loading pretrained parameters: {}".format(key))
            state_dict[key] = xlm_state_dict[key]
    return state_dict

