#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import logging

logger = logging.getLogger(__name__)


def sequence_ctc_loss_with_logits(
    logits: torch.FloatTensor,
    logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
    targets: torch.LongTensor,
    target_mask: Union[torch.FloatTensor, torch.BoolTensor],
    blank_index: torch.LongTensor,
    label_smoothing=0,
    reduction='mean',  # or batch_sum
) -> torch.FloatTensor:

    # lengths : (batch_size, )
    # calculated by counting number of mask
    logit_lengths = (logit_mask.bool()).long().sum(1)
    target_lengths = (target_mask.bool()).long().sum(1)

    # (batch_size, T, n_class)
    log_probs = logits.log_softmax(-1)
    # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
    log_probs_T = log_probs.transpose(0, 1)

    #     assert (target_lengths == 0).any()
    targets = targets.long()
    targets = targets[target_mask.bool()]

    loss = F.ctc_loss(
        log_probs_T.float(),  # compatible with fp16
        targets,
        logit_lengths,
        target_lengths,
        blank=blank_index,
        reduction="none" if reduction == 'batch_sum' else 'mean',
        zero_infinity=True,
    )

    if reduction == 'batch_sum':
        return loss

    # n_invalid_samples = (logit_lengths < target_lengths).long().sum()
    # if n_invalid_samples > 0:
    #     logger.warning(
    #         f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
    #     )
    #     raise Exception

    if label_smoothing > 0:
        # n_vocob = log_probs.size(-1)
        # kl_loss = F.kl_div(log_probs, torch.full_like(log_probs, 1/n_vocob), reduction='none', log_target=False).sum(-1)
        # # kl_loss = log_probs.neg().sum(-1) / n_vocob
        # kl_loss = ((kl_loss * logit_mask.float()).sum(-1) / logit_lengths)[logit_lengths >= target_lengths].mean()
        
        smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss

    return loss


@register_criterion("my_nat_loss")
class LabelSmoothedDualImitationCriterion2(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss * factor, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample.get("prev_target", None)

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", _losses.get("loss", 0.0))]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar("loss", loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
