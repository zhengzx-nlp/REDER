# -*- coding: UTF-8 -*- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- BOW (or word prediction loss)------------------------- #
def convert_to_past_labels(labels, padding_idx):
    """
    Args:
        labels: [bsz, tlen]

    Returns:
        [bsz, tlen, tlen].
    """
    tlen = labels.size(1)
    padded_mask = labels.eq(padding_idx)

    # use upper triangle to masking in ascending manner
    # [tlen+1, tlen+1]
    # including current one itself as one of
    _upper = torch.triu(padded_mask.new_ones(tlen, tlen), 0).bool()
    # the past tokens (1) or not (0) leads to
    # similar results.
    # [bsz, tlen+1, tlen+1]
    mask = _upper[None, :, :] + padded_mask[:, None, :]

    # [bsz, tlen+1, tlen+1]
    past_labels = labels[:, None, :].repeat(1, tlen, 1)
    past_labels.masked_fill_(mask, padding_idx)

    return past_labels


def convert_to_future_labels(labels, padding_idx):
    """
    Args:
        labels: [bsz, tlen+1]

    Returns:
        [bsz, tlen, tlen+1].
    """
    tlen = labels.size(1)
    padded_mask = labels.eq(padding_idx)

    # use lower triangle to masking in decending manner
    # [tlen+1, tlen+1]
    # including current one itself as one of
    _lower = torch.tril(padded_mask.new_ones(tlen, tlen), 0).bool()
    # the future tokens (-1) or not (0) leads
    # to similar results.
    # [bsz, tlen+1, tlen+1]
    mask = _lower[None, :, :] + padded_mask[:, None, :]

    # [bsz, tlen+1, tlen+1]
    future_labels = labels[:, None, :].repeat(1, tlen, 1)
    future_labels.masked_fill_(mask, padding_idx)

    return future_labels


def word_prediction_loss(logprobs_past, logprobs_future, labels, padding_idx, valid_mask=None):
    """ Word prediction loss, or Bag-of-Word loss, adapted from
    Weng, et al., "Neural Machine Translation with Word Predictions".
    <https://www.aclweb.org/anthology/D17-1013/>
    
    Args:
        logprobs_past: [bsz, tlen, nwords]
        logprobs_future: [bsz, tlen, nwords]
        labels: [bsz, tlen+1].
            This sequence of labels should include <BOS> and <EOS>,
            so the length would be `tlen+1`

    Returns:
        Tuple[Tensor]. Word prediction losses for past and future.
            ([bsz,], [bsz])
    """
    bsz, tlen, nwords = logprobs_past.size()

    # [bsz, tlen, tlen+1]
    past_labels = convert_to_past_labels(labels, padding_idx)
    future_labels = convert_to_future_labels(labels, padding_idx)
    # [bsz, tlen] 0 for padding, 1 for valid
    mask = labels.ne(padding_idx).float()
    if valid_mask is not None:
        mask = mask * valid_mask.float()
    # [bsz]
    lens = mask.sum(-1)

    def _compute_loss(_logp, _label):
        # [bsz, tlen, tlen+1]
        _loss = torch.gather(_logp, -1, _label)

        # average at every step
        # [bsz, tlen, tlen]
        _mask = _label.ne(padding_idx).float()
        # [bsz, tlen]
        _step_nums = _mask.sum(-1)
        _loss = (_loss * _mask).sum(-1) / (_step_nums + 1e-4)

        # average over all steps
        # [bsz,]
        _loss = (_loss * mask).sum(-1) / lens
        return _loss.mean() # [1]

    # [bsz]. loss for each sentence in a batch
    wploss_past = _compute_loss(-logprobs_past, past_labels)
    wploss_future = _compute_loss(-logprobs_future, future_labels)

    return wploss_past, wploss_future


# ------------------------- BCA ------------------------- #
def get_average(mask):
    mask = mask.float()
    scores = mask / mask.sum(-1, keepdim=True)
    scores = torch.where(torch.isnan(scores),
                         torch.zeros_like(scores),
                         scores)
    return scores


def get_prev_sequence_average(tensor):
    b, l, d = tensor.size()

    # -1: including current one itself
    # 0: excluding current one itself
    mask = tensor.new_tensor(torch.tril(torch.ones([l, l]), -1))
    mask = get_average(mask)
    mask = mask[None, :, :].repeat(b, 1, 1)

    # [b, l, l] x [b, l, d] -> [b, l, d]
    out = mask @ tensor
    return out


def get_sub_sequence_average(tensor):
    b, l, d = tensor.size()

    # 0: including current one itself
    # 1: excluding current one itself
    mask = tensor.new_tensor(torch.triu(torch.ones([l, l]), 0))
    mask = get_average(mask)
    mask = mask[None, :, :].repeat(b, 1, 1)

    # [b, l, l] x [b, l, d] -> [b, l, d]
    out = mask @ tensor
    return out


def bilingual_content_alignment_loss(
        decoder_hiddens,
        past_capsules,
        future_capsules,
        labels):
    """ 
    Args:
        decoder_hiddens: [bsz, tlen, d]
        past_capsules: [bsz, tlen, d]
        future_capsules: [bsz, tlen, d]
        labels: [bsz, tlen] (excluding <BOS>)

    Returns:
        Tuple[Tensor]. bca losses for past and future.
            ([bsz,], [bsz])
    """
    nonpadded_mask = labels.ne(PAD)  # [bsz, tlen]
    lens = nonpadded_mask.sum(-1)  # [bsz]

    def _compute_loss(_input, _target):
        # [bsz, tlen, d]
        _loss = F.mse_loss(
            _input,
            _target.detach(),
            size_average=False,
            reduce=False)

        # average at every step
        # [bsz, tlen]
        _loss = _loss.mean(-1)

        # average over all steps
        # [bsz]
        _loss = (_loss * nonpadded_mask).sum(-1) / lens
        return _loss

    bca_loss_past = _compute_loss(
        past_capsules,
        get_prev_sequence_average(decoder_hiddens))
    bca_loss_future = _compute_loss(
        future_capsules,
        get_sub_sequence_average(decoder_hiddens))

    return bca_loss_past, bca_loss_future

