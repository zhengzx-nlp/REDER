# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch._C import dtype
from fairseq.data.dictionary import Dictionary
import os
import logging

import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset, data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def _swap_direction(sample):
    sample["net_input"]["src_tokens"], sample["target"] = sample["target"], sample["net_input"]["src_tokens"]

@register_task('translation_nat')
class TranslationNATTask(TranslationTask):
    """
    Modified from fairseq.tasks.translation_lev

    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.dataset_cls = LanguagePairDataset
        self.training = False

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask', 'schedule_random_mask'])
        parser.add_argument(
            '--noise-ratio-range',
            default='0.5,1.0')
        parser.add_argument(
            '--noise-end-update', type=int,
            default=150_000)
        parser.add_argument(
            "--decode-with-oracle-length",
            action="store_true",
            help="use normal transformer decoder",
        )
        parser.add_argument(
            "--center-aligned-dataset",
            action="store_true",
            help="make pair dataset center aligned",
        )
        parser.add_argument(
            "--valid-reverse-model",
            action="store_true",
            default=False,
            help="use reverse model for valid",
        )    
        parser.add_argument(
            "--use-reverse-model",
            action="store_true",
            default=False,
            help="always use reverse model",
        )   
        parser.add_argument(
            "--no-length-filtering",
            action="store_true",
            default=False,
            help="do not filter examples with invalid length",
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = False  # utils.eval_bool(args.left_pad_source)
        args.left_pad_target = False  # utils.eval_bool(args.left_pad_target))

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(args, os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(args, os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, args, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """

        d = Dictionary.load(filename)
        if hasattr(args, "ctc_loss"):
            d.blank_index = d.add_symbol("<blank>")
            d.nspecial += 1
            d.blank = d.blank_index
        return d

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        original_size = len(indices)
        if ignore_invalid_inputs and hasattr(self.args, "ctc_loss") \
            and (not getattr(self.args, "no_length_filtering", False)):
            max_positions = (
                (dataset.tgt_sizes[indices] * self.args.upsampling).tolist(),
                (dataset.src_sizes[indices] * self.args.upsampling).tolist(),
            )
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception((
                    'Size of sample #{} is invalid (={}) since max_positions={}, '
                    'skip this example with --skip-invalid-size-inputs-valid-test'
                ).format(ignored[0], dataset.size(ignored[0]), max_positions))
            if hasattr(self.args, "ctc_loss"):
                logger.warning((
                    'when ctc loss enabled, {} samples have invalid sizes and will be skipped, '
                    'where the src_len * {} < tgt_len'
                ).format(len(ignored), self.args.upsampling))
            else:
                logger.warning((
                    '{} samples have invalid sizes and will be skipped, '
                    'max_positions={}, first few sample ids={}'
                ).format(len(ignored), max_positions, ignored[:10]))
                
            logger.info(f"Dataset original size: {original_size}, filtered size: {len(indices)}")

        return indices

    @property
    def _noise_ratio_range(self):
        start, end = self.args.noise_ratio_range.split(',')
        return float(start), float(end)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
            dataset_cls=self.dataset_cls
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError("Constrained decoding with the translation_lev task is not supported")

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def inject_noise(self, target_tokens, update_num=None, training=True):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                target_tokens.ne(bos) & \
                target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            # make sure to mask at least one token.
            target_length = target_length + 1

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(
                target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        def _schedule_random_mask(target_tokens, update_num):
            if not training:
                return _full_mask(target_tokens)

            def _get_inject_noise_ratio(update_num):
                if update_num > self.args.noise_end_update:
                    return 1.0
                start, end = self._noise_ratio_range
                ratio = start + (end - start) * update_num / (self.args.noise_end_update)
                return ratio

            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                target_tokens.ne(bos) & \
                target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()

            noise_ratio = _get_inject_noise_ratio(update_num)

            target_length = target_length * noise_ratio
            # make sure to mask at least one token.
            target_length = target_length + 1

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(
                target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _bert_random_mask(target_tokens):
            mask_prob = 0.15
            leave_unmasked_prob = 0.1
            random_replace_prob = 0.1            

            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                target_tokens.ne(bos) & \
                target_tokens.ne(eos)
            
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()

            # noise_ratio = _get_inject_noise_ratio(update_num)

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
                rand_or_unmask = mask & (torch.rand(sz, device=mask.device).float() < rand_or_unmask_prob)
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
                    .random_(self.tgt_dict.nspecials, len(self.tgt_dict))

            return prev_target_tokens

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        elif self.args.noise == 'schedule_random_mask':
            return _schedule_random_mask(target_tokens, update_num)
        elif self.args.noise == 'bert_mask':
            return _bert_random_mask(target_tokens)
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import \
            IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 0),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(
                args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        sample['prev_target'] = self.inject_noise(sample['target'], update_num=update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if getattr(model, "reverse", False):
                _swap_direction(sample)
            sample['prev_target'] = self.inject_noise(sample['target'], training=False)
            loss, sample_size, logging_output = criterion(model, sample)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(
                self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, dict, escape_unk=False):
            extra_symbols_to_ignore = []
            if hasattr(dict, "blank_index"): extra_symbols_to_ignore.append(dict.blank_index)
            if hasattr(dict, "mask_index"): extra_symbols_to_ignore.append(dict.mask_index)
            s = dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
                extra_symbols_to_ignore=extra_symbols_to_ignore or None
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs, srcs = [], [], []
        for i in range(len(gen_out)):
            hyp = gen_out[i][0]['tokens']
            if hasattr(self.args, "ctc_loss"):
                _toks = hyp.int().tolist()
                hyp = hyp.new_tensor([v for i, v in enumerate(_toks) if i == 0 or v != _toks[i-1]])
            
            hyps.append(decode(hyp, self.tgt_dict))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                # sample['target'][i],
                self.tgt_dict,
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            srcs.append(decode(
                utils.strip_pad(sample["net_input"]["src_tokens"][i], self.src_dict.pad()),
                # sample["net_input"]["src_tokens"][i],
                self.src_dict,
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example source    : ' + srcs[0])
            logger.info('example reference : ' + refs[0])
            logger.info('example hypothesis: ' + hyps[0])

        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
