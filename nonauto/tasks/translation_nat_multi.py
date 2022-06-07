# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os
from argparse import Namespace
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from fairseq import metrics, utils
from fairseq.data import (AppendTokenDataset, ConcatDataset,
                          LanguagePairDataset, PrependTokenDataset,
                          RoundRobinZipDatasets, StripTokenDataset,
                          TransformEosLangPairDataset, TruncateDataset,
                          data_utils, encoders, indexed_dataset)
from fairseq.data.dictionary import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange
from nonauto.data.simple_language_pair_dataset import SimpleLanguagePairDataset
from nonauto.tasks.translation_nat import TranslationNATTask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def get_lang_pair(s, t):
    return f"{s}-{t}"


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return id


@contextlib.contextmanager
def set_model_lang_pair(model, lang_pair):
    old_value = model.eval_lang_pair
    model.eval_lang_pair = lang_pair
    yield
    model.eval_lang_pair = old_value


@register_task('translation_nat_multi')
class TranslationNATMultiTask(TranslationNATTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    # def __init__(self, args, src_dict, tgt_dict):
    #     super().__init__(args, src_dict, tgt_dict)
    #     self.direction_lang_pairs = {
    #         "forward": get_lang_pair(args.source_lang, args.target_lang),
    #         "reverse": get_lang_pair(args.target_lang, args.source_lang),
    #     }
    #     if args.training_forward_direction_only:
    #         self.direction_lang_pairs.pop("reverse")

    #     self.best_bleus = {lp: 0.0 for _, lp in self.direction_lang_pairs.items()}

    def __init__(self, args, dicts, training):
        super(TranslationTask, self).__init__(args)
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
            self.is_inference = False
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
            self.is_inference = True
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs + (args.extra_eval_lang_pairs if args.extra_eval_lang_pairs else [])
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        
        self.langs = list(dicts.keys())

        self.best_bleus = {lp: 0.0 for lp in self.eval_lang_pairs}

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationNATTask.add_args(parser)
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('--extra-eval-lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of EXTRA language pairs for validation: en-de,en-fr,de-fr')

        # reserved for multilingual setting but not used right now
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')

        # decoding 
        parser.add_argument('--ctc-decode-with-beam', default=1, type=int,
                            help='use ctc beam search')
        parser.add_argument('--ctc-decode-score-reconstruction-weight', default=0, type=float,
                            help='weight for reconstruction score when decoding with CTC.')
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kargs):    
        args.left_pad_source = False  # utils.eval_bool(args.left_pad_source)
        args.left_pad_target = False  # utils.eval_bool(args.left_pad_target)

        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        if isinstance(args.lang_pairs, str):
            args.lang_pairs = args.lang_pairs.split(',')
        if args.extra_eval_lang_pairs and isinstance(args.extra_eval_lang_pairs, str):
            args.extra_eval_lang_pairs = args.extra_eval_lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dicts[lang] = cls.load_dictionary(args, os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(lang_to_add))
            logger.info('[{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training

    @classmethod
    def load_dictionary(cls, args, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """

        d = Dictionary()
        d.add_from_file(filename)
        if hasattr(args, "ctc_loss"):
            d.blank_index = d.add_symbol("<blank>")
            d.blank = d.blank_index

        return d

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
           and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a dataset split."""
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # TODO: filter pairs with invalid length
        def filter_indeces_by_sizes(dataset):
            indices = dataset.ordered_indices()
            max_positions = (
                self.args.max_source_positions,
                (dataset.src_sizes[indices] * self.args.upsampling).tolist(),
            )
            indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
            

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            # return langpair_dataset
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        lang_pairs = self.model_lang_pairs if split == 'train' else self.eval_lang_pairs
        datasets = OrderedDict()
        for lang_pair in lang_pairs:
            ds = language_pair_dataset(lang_pair)
            if ds is not None:
                datasets[lang_pair] = ds
        self.datasets[split] = RoundRobinZipDatasets(
            datasets,
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        return indices

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError("Constrained decoding with the translation_lev task is not supported")

        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        def check_args():
            messages = []
            if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
                messages.append('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append('--encoder-langtok should be {}.'.format(args.encoder_langtok))
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append('--decoder-langtok should {} be set.'.format("" if args.decoder_langtok else "not"))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        # if not isinstance(model, FairseqMultiModel):
        #     raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')

        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))

        return model

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import \
            IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=0,
            beam_size=getattr(args, 'ctc_decode_with_beam', 1),
            reranking=getattr(
                args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=False,  # not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            ctc_model=True
        )

    def _maybe_enable_auxiliary_losses(self, model, update_num):
        if (not model.enable_fba_loss) and self.args.fba_loss and update_num >= self.args.enable_fba_after_update:
            model.enable_fba_loss = True
        if (not model.enable_cycle_loss) and self.args.cycle_loss and update_num >= self.args.enable_cycle_loss_after_update:
            model.enable_cycle_loss = True

    def _per_direction_train_step(self, model, criterion, sample, lang_pair, optimizer, ignore_grad, update_num):
        # sample['prev_target'] = self.inject_noise(sample['target'], update_num=update_num)
        self._maybe_enable_auxiliary_losses(model, update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        from collections import defaultdict
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
        curr_lang_pairs = [
            lang_pair
            for lang_pair in self.model_lang_pairs
            if sample[lang_pair] is not None and len(sample[lang_pair]) != 0
        ]

        for idx, lang_pair in enumerate(curr_lang_pairs):
            with model.set_lang_pair(lang_pair):
                loss, sample_size, logging_output = self._per_direction_train_step(
                    model,
                    criterion,
                    sample[lang_pair],
                    lang_pair,
                    optimizer,
                    ignore_grad,
                    update_num=update_num,
                )
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            # agg_sample_size += sample_size
            agg_sample_size = 1
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def _per_direction_valid_step(self, model, criterion, sample, lang_pair):
        loss, sample_size, logging_output = criterion(model, sample)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(
                self.sequence_generator, sample, model, lang_pair=lang_pair)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)

            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    flag = 0.
                else:
                    flag = 1.
                    with model.set_lang_pair(lang_pair):
                        loss, sample_size, logging_output = self._per_direction_valid_step(
                            model, criterion, sample[lang_pair], lang_pair=lang_pair)

                    agg_loss += loss.data.item()
                    # TODO make summing of the sample sizes configurable
                    # agg_sample_size += sample_size
                    agg_sample_size = 1
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k] * flag
                    agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k] * flag
        return agg_loss, agg_sample_size, agg_logging_output

    def _inference_with_bleu(self, generator, sample, model, lang_pair):
        import sacrebleu

        def decode(toks, dict, escape_unk=False):
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
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs, srcs = [], [], []
        for i in range(len(gen_out)):
            hyp = gen_out[i][0]['tokens']
            hyps.append(decode(hyp, self.target_dictionary))

            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.target_dictionary.pad()),
                # sample['target'][i],
                self.target_dictionary,
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            srcs.append(decode(
                utils.strip_pad(sample["net_input"]["src_tokens"][i], self.source_dictionary.pad()),
                # sample["net_input"]["src_tokens"][i],
                self.source_dictionary,
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))

        if self.args.eval_bleu_print_samples:
            logger.info(f'[{lang_pair}] example source    : ' + srcs[0])
            logger.info(f'[{lang_pair}] example reference : ' + refs[0])
            logger.info(f'[{lang_pair}] example hypothesis: ' + hyps[0])

        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        if self.is_inference:
            models[0]._eval_lang_pair = self.eval_lang_pairs[0]

        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)

    def reduce_metrics(self, logging_outputs, criterion):
        super(TranslationTask, self).reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu and not criterion.training:

            def sum_logs(key):
                s = sum(log.get(key, 0) for log in logging_outputs)
                if isinstance(s, torch.Tensor):
                    s = s.item()
                return s

            def reduce_bleu_per_direction(prefix):
                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs(prefix + '_bleu_counts_' + str(i)))
                    totals.append(sum_logs(prefix + '_bleu_totals_' + str(i)))

                # if max(totals) > -1:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar(f'_{prefix}_bleu_counts', np.array(counts))
                metrics.log_scalar(f'_{prefix}_bleu_totals', np.array(totals))
                metrics.log_scalar(f'_{prefix}_bleu_sys_len', sum_logs(prefix + '_bleu_sys_len'))
                metrics.log_scalar(f'_{prefix}_bleu_ref_len', sum_logs(prefix + '_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect

                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters[f'_{prefix}_bleu_counts'].sum,
                        total=meters[f'_{prefix}_bleu_totals'].sum,
                        sys_len=meters[f'_{prefix}_bleu_sys_len'].sum,
                        ref_len=meters[f'_{prefix}_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived(prefix + 'bleu', compute_bleu)
                # _bleu = metrics.get_smoothed_value("valid", prefix + 'bleu')

                    # metrics.log_scalar('bleu', _bleu)

                    # best_bleu = max(_bleu, self.best_bleus[prefix[:-1]])
                    # metrics.log_scalar(prefix + 'best_bleu', best_bleu)
                # return _bleu

            # total_bleu = 0.0
            for lang_pair in self.eval_lang_pairs:
                reduce_bleu_per_direction(prefix=f"{lang_pair}:")
                # total_bleu += _bleu
            # metrics.log_derived('bleu', total_bleu / len(self.direction_lang_pairs))

            def compute_bleu(meters):
                bleus = []
                for lang_pair in self.eval_lang_pairs:
                    bleu = 0.0
                    if f'{lang_pair}:bleu' in meters:
                        bleu = meters[f'{lang_pair}:bleu'].fn(meters)
                    bleus.append(bleu)
                return np.mean(bleus)
            metrics.log_derived("bleu", compute_bleu)

    @property
    def source_dictionary(self):
        return next(iter(self.dicts.values()))

    @property
    def target_dictionary(self):
        return next(iter(self.dicts.values()))
