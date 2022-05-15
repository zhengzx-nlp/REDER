import contextlib
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (FairseqMultiModel, register_model,
                            register_model_architecture)
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel
from fairseq.models.transformer import (Embedding, Linear, TransformerDecoder,
                                        TransformerEncoder)
from fairseq.modules import FairseqDropout, LayerNorm
from nonauto.criterions.nat_loss import sequence_ctc_loss_with_logits
from nonauto.modules.gradrev import GradientReversal
from nonauto.modules.position_embedding import RelativePositionEmbedding
from nonauto.modules.reversible_utils import *

from .layers import *
from .model_utils import *

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


logger = logging.getLogger(__name__)


@register_model("mREDER")
class MREDER(NATransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Univseral Tranformer
        parser.add_argument('--encoder-cross-layer-sharing', action='store_true',
                            help='sharing parameters for encoder layers')

        # REDER specific
        parser.add_argument(
            "--apply-bert-init", action="store_true",
            help="use custom param initialization for BERT")
        parser.add_argument(
            "--self-attention-type", default="normal", choices=["normal", "relative"],
            help="self attention type")
        parser.add_argument(
            "--out-norm-type", type=str, default="actnorm", choices=["layernorm", "actnorm"],
            help="how to perform feature normalization for decoder output")
        parser.add_argument(
            "--upsampling", type=int, metavar='N', default=1,
            help="upsampling ratio")
        parser.add_argument(
            "--ctc-loss", action="store_true", default=False,
            help="enable fba loss")
        parser.add_argument(
            "--fba-loss", action="store_true", default=False,
            help="enable fba loss")
        parser.add_argument(
            "--enable-fba-loss-after-update", type=int, metavar='N', default=100_000,
            help="enable fba loss after some predefined updates")
        parser.add_argument(
            "--cycle-loss", action="store_true", default=False,
            help="enable cycle loss")
        parser.add_argument(
            "--enable-cycle-loss-after-update", type=int, metavar='N', default=100_000,
            help="enable cycle loss after some predefined updates")
        parser.add_argument(
            "--lang-adv-loss", action="store_true", default=False,
            help="enable language adversarial loss")
        parser.add_argument(
            "--layer-norm-type", type=str, default="prenorm", choices=["prenorm", "sandwich"],
            help="layer norm type")
        parser.add_argument(
            "--pretrained-checkpoint", type=str,
            help="pretrained model checkpoint path.")
        parser.add_argument(
            "--share-all", action="store_true", default=False,
            help="share model parameters for all languages")
        parser.add_argument(
            "--lang-end-inner-order", type=str, choices=['fs', 'sf'], default='sf',
            help="order of operations of each layer, attention->FFN or FFN->attention ")
        parser.add_argument(
            "--split-embedding", action="store_true", default=False,
            help="double the dimension of embedding so as not to copy embeddings before feeding to RevNet")
        parser.add_argument(
            "--feature-shuffle", action="store_true", default=False,
            help="shuffle feature dimension-wise between each RevNet layer.")

    def __init__(self, args, encoder, decoder, langs):
        super().__init__(args, encoder, decoder)
        delattr(self, 'decoder')

        self.args = args
        self.langs = langs

        self._eval_lang_pair = None
        self._reversed = False

        self.enable_fba_loss = False
        self.enable_cycle_loss = False

        self._build_dictionaries_and_embeds()
        self._build_output_projections()
        self._build_fba_predictors()
        self._build_lang_discriminator()

        if hasattr(args, "pretrained_checkpoint"):
            pretrained_loaded_state_dict = upgrade_state_dict_with_pretrained_weights(
                state_dict=self.state_dict(),
                pretrained_checkpoint=args.pretrained_checkpoint,
            )
            self.load_state_dict(pretrained_loaded_state_dict, strict=False)

    def _build_dictionaries_and_embeds(self):
        self.dicts = {
            lang: self.encoder.dictionary for lang in self.langs
        }
        self.embed_tokens = nn.ModuleDict({
            lang: self.encoder.embed_tokens for lang in self.langs
        })
        self.embed_positions = nn.ModuleDict({
            lang: self.encoder.embed_positions for lang in self.langs
        })
        self.embed_scale = 1.0 if self.args.no_scale_embedding else math.sqrt(
            self.args.encoder_embed_dim)
        self.dropout_module = FairseqDropout(
            self.args.dropout, module_name=self.__class__.__name__)

        self.blank_index = getattr(
            self.dicts[self.langs[0]], "blank_index", None)

    def _build_output_projections(self):
        def _build(weight):
            output_projection = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=False,
            )
            output_projection.weight = weight
            return output_projection

        self.output_projections = nn.ModuleDict({
            lang: _build(self.embed_tokens[lang].weight) for lang in self.langs
        })

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.split_embedding:
            args.embed_dim = args.encoder_embed_dim * 2

        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:

            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=getattr(args, "embed_dim", args.encoder_embed_dim),
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            raise ValueError(f"{cls.__name__} requires --share-all-embeddings")

        shared_dict = task.dicts[task.langs[0]]
        encoder = cls._build_encoder(
            args, task, shared_dict, shared_encoder_embed_tokens)
        decoder = SimpleDecoder(
            args, shared_dict, shared_decoder_embed_tokens
        )

        return cls(args, encoder, decoder, task.langs)

    @classmethod
    def _build_encoder(cls, args, task, dicts, embed_tokens):
        encoder = RevTransformerEncoder(
            args, dicts, embed_tokens, task.langs) 
        return encoder

    def _make_padding_mask(self, tokens, pad_mask_token=True):
        padding_mask = tokens.eq(self.pad)
        return padding_mask

    def _convert_mask_to_lengths(self, mask):
        return mask.long().sum(-1)

    def _convert_lengths_to_mask(self, lens, maxlen=None):
        # lens: (bsz)
        maxlen = maxlen or lens.max()
        lens = lens.view(-1)
        mask = torch.arange(maxlen, device=lens.device)[
            None, :] < lens[:, None]
        return mask

    @contextlib.contextmanager
    def reverse(self):
        old_value = self._reversed
        self._reversed = not old_value
        yield
        self._reversed = old_value

    @contextlib.contextmanager
    def set_lang_pair(self, lang_pair):
        old_value = self._eval_lang_pair
        self._eval_lang_pair = lang_pair
        yield
        self._eval_lang_pair = old_value

    @property
    def lang_pair(self):
        if not self._eval_lang_pair:
            raise ValueError(
                f"{self.__class__.__name__}._eval_lang_pair should be assigned a value.")
        src, tgt = self._eval_lang_pair.split('-')
        if self._reversed:
            src, tgt = tgt, src
        return src, tgt
        # return (src, tgt) if not self._reverse else (tgt, src)

    def _forward(self, src_tokens, tgt_tokens, feature_only=False, no_upsample=False, reduction='mean', **kwargs):
        # maybe upsample for ctc
        if not no_upsample:
            src_tokens = self._maybe_upsample(src_tokens)

        inp, emb, padding_mask = self._embed(src_tokens)

        encoder_out = self._encode(inp, emb, padding_mask)

        logits = self._decode(encoder_out, normalize=False)

        if feature_only:
            return None, {"encoder_out": encoder_out, "logits": logits}

        # masks
        logit_mask = ~self._make_padding_mask(src_tokens)
        word_pred_mask = ~self._make_padding_mask(tgt_tokens)

        # ctc loss
        ctc_loss = sequence_ctc_loss_with_logits(
            logits=logits,
            logit_mask=logit_mask,
            targets=tgt_tokens,
            target_mask=word_pred_mask,
            blank_index=self.blank_index,
            label_smoothing=self.args.label_smoothing,
            reduction=reduction
        )

        net_output = {
            "word_pred_ctc": {
                "loss": ctc_loss,
                "nll_loss": True,
            },
        }
        return net_output, {"encoder_out": encoder_out, "logits": logits}

    def _embed(self, tokens, lang=None):
        if not lang:
            sl, tl = self.lang_pair
            lang = sl

        x = embed = self.embed_scale * self.embed_tokens[lang](tokens)
        if self.embed_positions[lang] is not None:
            x = embed + self.embed_positions[lang](tokens)
        x = self.dropout_module(x)

        padding_mask = self._make_padding_mask(tokens) 

        return x, embed, padding_mask

    def _encode(self, inp, emb, padding_mask):
        sl, tl = self.lang_pair

        encoder_out = self.encoder.forward(
            inp, emb, padding_mask,
            src_lang=sl, tgt_lang=tl,
            return_all_hiddens=self.args.fba_loss
        )

        return encoder_out

    def _decode(self, encoder_out, normalize=False, reverse=False, prev_output_tokens=None, step=0):
        sl, tl = self.lang_pair

        feature = encoder_out.encoder_out
        logits = self.output_projections[tl](feature)
        return F.log_softmax(logits, -1) if normalize else logits

    def _maybe_upsample(self, tokens):
        if self.args.upsampling <= 1:
            return tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x
        return _us(tokens, self.args.upsampling)

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None, tgt_tokens=None, **kwargs):
        return self.forward_unidir(src_tokens, tgt_tokens, **kwargs)

    def forward_unidir(self, src_tokens, tgt_tokens, **kwargs):
        net_output, extras = self._forward(
            src_tokens=src_tokens, tgt_tokens=tgt_tokens)

        if self.args.fba_loss and self.enable_fba_loss:
            fba_loss = self._compute_fba_loss(
                src_tokens, tgt_tokens, extras["logits"], extras["encoder_out"])
            net_output["fba_cos"] = {"loss": fba_loss, "factor": 0.5}

        if self.args.cycle_loss and self.enable_cycle_loss:
            cycle_loss_net_output = self._compute_cycle_loss(
                src_tokens, extras["logits"], extras["encoder_out"].encoder_padding_mask)
            net_output["cycle_ctc"] = {"factor": 0.5, **cycle_loss_net_output}

        if self.args.lang_adv_loss:
            lang_adv_out = self._compute_lang_adv_loss(extras["encoder_out"])
            net_output["lang_adv"] = {"factor": 0.1, **lang_adv_out}
        return net_output

    # forward-backward agreement loss #
    def _build_fba_predictors(self):
        if self.args.fba_loss:
            self.fba_loss_fn = cos_distance_loss
            embed_dim = self.args.encoder_embed_dim
            # self.predictors = nn.ModuleList([
            #     BottleneckFFN(embed_dim * 2, embed_dim // 2, dropout=0.0)
            #     for _ in range(self.args.encoder_layers*2+1)
            # ])
            self.predictors = nn.ModuleList([
                nn.Identity()
                for _ in range(self.args.encoder_layers*2+1)
            ])

    def _representation_predict(self, hiddens, predict=False):
        if not predict or not hasattr(self, "predictors"):
            return hiddens
        preds = []
        for pp, hh in zip(self.predictors, hiddens):
            preds.append(pp(hh))
        return preds

    def _compute_fba_loss(self, src_tokens, tgt_tokens, logits, forward_encoder_out):
        fwd_states, bwd_states, mask = self._prepare_fba_loss(
            src_tokens, tgt_tokens, logits, forward_encoder_out)

        fwd_states_pred = self._representation_predict(fwd_states, predict=True)
        bwd_states = self._representation_predict(bwd_states, predict=False)

        fwd_states_pred = torch.stack(fwd_states_pred, 2)
        bwd_states = torch.stack(bwd_states, 2)

        fba_loss = self.fba_loss_fn(
            output=fwd_states_pred,
            target=bwd_states.detach(),
            # target=bwd_states,
            mask=mask,
        )
        return fba_loss

    def _prepare_fba_loss(self, src_tokens, tgt_tokens, logits, forward_encoder_out):
        src_padding_mask = forward_encoder_out.encoder_padding_mask
        tgt_padding_mask = self._make_padding_mask(tgt_tokens)

        src_lengths = self._convert_mask_to_lengths(~src_padding_mask)
        tgt_lengths = self._convert_mask_to_lengths(~tgt_padding_mask)

        fwd_states = forward_encoder_out.encoder_states

        from nonauto.modules.ctc_utils import convert_alignment_to_symbol
        from torch_imputer import best_alignment

        log_prob = F.log_softmax(logits, dim=-1)
        best_aligns = best_alignment(
            log_prob.transpose(0, 1).float(),
            tgt_tokens, src_lengths, tgt_lengths, self.blank_index, True)
        aligned_tgt_tokens = convert_alignment_to_symbol(
            best_aligns, tgt_tokens, self.blank_index, src_padding_mask, self.pad)
        # aligned_tgt_tokens = self._maybe_upsample(src_tokens)

        self.encoder.reuse_seed(True)

        with self.reverse():
            _, bwd_extra = self._forward(
                src_tokens=aligned_tgt_tokens,
                tgt_tokens=None,
                feature_only=True,
                no_upsample=True
            )
        bwd_states = bwd_extra["encoder_out"].encoder_states[::-1]

        self.encoder.reuse_seed(False)

        mask = ~src_padding_mask

        return fwd_states, bwd_states, mask        
    
    # cycle consistency loss #
    def _compute_cycle_loss(self, src_tokens, logits, src_padding_mask):
        decoded_tokens = self._prepare_cycle_loss(logits, src_padding_mask)

        with self.reverse():
            rev_net_output, _ = self._forward(
                src_tokens=decoded_tokens,
                tgt_tokens=src_tokens,
            )
        return rev_net_output["word_pred_ctc"]

    def _prepare_cycle_loss(self, logits, src_padding_mask):
        _, batch_ctc_decoded_tokens = F.log_softmax(logits, -1).max(-1)

        from nonauto.modules.ctc_utils import \
            post_process_ctc_decoded_tokens_with_pad
        decoded_tokens = post_process_ctc_decoded_tokens_with_pad(
            batch_ctc_decoded_tokens,
            self.pad, left_pad=False
        )
        return decoded_tokens

    def _build_lang_discriminator(self):
        if self.args.lang_adv_loss:
            self.lang_dict = {self.langs[i]: i for i in range(len(self.langs))}
            embed_dim = self.args.encoder_embed_dim

            layers = [GradientReversal(),
                      nn.Dropout(0.2),
                      nn.Linear(embed_dim*2, embed_dim)]
            for i in range(getattr(self.args, 'lang_dis_layers', 2)):
                layers.append(nn.Linear(embed_dim, embed_dim))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.2))
            layers.append(nn.Linear(embed_dim, len(self.langs)))

            self.lang_dis = nn.Sequential(*layers)

    def _compute_lang_adv_loss(self, encoder_out):
        sl, tl = self.lang_pair

        # [bsz, L, 2d]
        mid = encoder_out.encoder_mid
        mask = ~encoder_out.encoder_padding_mask
        mid_avg = tensor_mean_by_mask(mid, mask)

        # [bsz, #lang]
        logits = self.lang_dis(mid_avg)

        bsz = logits.size(0)
        lang_labels = logits.new_full(
            (bsz, ), self.lang_dict[sl], dtype=torch.int64)

        return {
            "out": logits, "tgt": lang_labels
        }

    ##### methods for decoding  ######
    def initialize_output_tokens(self, encoder_out, src_tokens, **kwargs):
        if hasattr(self.args, "ctc_loss"):
            initial_output_tokens = src_tokens.clone()  # [B, T]
        else:
            initial_output_tokens = kwargs["sample"]["target"].clone()

        initial_output_tokens = self._maybe_upsample(initial_output_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            src_tokens=src_tokens.clone(),
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def forward_encoder(self, encoder_inputs):
        tokens, lengths = encoder_inputs

        tokens = self._maybe_upsample(tokens)

        inp, emb, padding_mask = self._embed(tokens)

        encoder_out = self._encode(inp, emb, padding_mask)

        return encoder_out

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        src_tokens = decoder_out.src_tokens

        # execute the decoder
        output_masks = ~encoder_out.encoder_padding_mask
        log_probs = self._decode(
            encoder_out=encoder_out,
            normalize=True,
            step=step,
        )

        if getattr(self.args, 'ctc_decode_with_beam', 1) == 1:
            output_scores, output_tokens = log_probs.max(-1)
            output_tokens[~output_masks] = self.pad
            output_scores[~output_masks] = 0.
        else:
            output_tokens, output_scores = self.ctc_beamsearch(
                log_probs, output_masks,
                src_tokens=src_tokens,
                score_reconstruction_weight=getattr(
                    self.args, 'ctc_decode_score_reconstruction_weight', 0)
            )

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=None
        )

    def ctc_beamsearch(
        self, log_probs, probs_mask, src_tokens=None, score_reconstruction_weight=0
    ):
        def _get_ctcdecoder():
            if not hasattr(self, 'ctcdecoder'):
                sl, tl = self.lang_pair
                labels = self.dicts[tl].symbols

                import multiprocessing
                nproc = multiprocessing.cpu_count()

                from ctcdecode import CTCBeamDecoder
                ctcdecoder = CTCBeamDecoder(
                    labels,
                    model_path=None,
                    alpha=0,
                    beta=0,
                    cutoff_top_n=max(40, self.args.ctc_decode_with_beam),
                    cutoff_prob=1.0,
                    beam_width=self.args.ctc_decode_with_beam,
                    num_processes=nproc,
                    blank_id=self.blank_index,
                    log_probs_input=True
                )
                self.ctcdecoder = ctcdecoder
            return self.ctcdecoder

        decoder = _get_ctcdecoder()
        probs_lens = probs_mask.int().sum(-1)
        device = probs_lens.device

        # BATCHSIZE x N_BEAMS X N_TIMESTEPS
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(
            log_probs, probs_lens)

        bbsz = beam_results.size(0) * beam_results.size(1)
        beam_results = beam_results.to(device).long().view(bbsz, -1)
        beam_scores = beam_scores.to(device).view(bbsz)
        out_lens = out_lens.to(device).view(bbsz)

        beam_mask = self._convert_lengths_to_mask(
            out_lens, maxlen=beam_results.size(-1))
        beam_results = beam_results.masked_fill_(~beam_mask, self.pad)

        if score_reconstruction_weight > 0:
            rec_scores = self.score_beam_reconstruction(
                src_tokens, beam_results)
            beam_scores += score_reconstruction_weight * rec_scores

        beam_scores = - \
            (beam_scores/out_lens).unsqueeze(-1).expand_as(beam_results)

        return beam_results, beam_scores

    def score_beam_reconstruction(self, src_tokens, beam_results):
        old_ls, self.args.label_smoothing = self.args.label_smoothing, 0

        beam_size = self.args.ctc_decode_with_beam
        src_tokens_beam = src_tokens[:, None, :].repeat(
            1, beam_size, 1).view(-1, src_tokens.size(-1))

        with self.reverse():
            rev_net_output, _ = self._forward(
                src_tokens=beam_results,
                tgt_tokens=src_tokens_beam,
                reduction='batch_sum'
            )
        # bsz*beam
        rec_scores = rev_net_output['word_pred_ctc']['loss'].clone()

        del rev_net_output
        del _
        self.args.label_smoothing = old_ls

        return rec_scores

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def max_positions(self):
        return (self.args.max_source_positions, self.args.max_target_positions)


class RevTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, langs):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args
        self.embed_dim = embed_tokens.embedding_dim
        self.langs = langs

        self.build_ends()
        self.build_out_norm()

    def reuse_seed(self, value):
        for lang, end in self.ends.items():
            for layer in end.layers:
                layer.reuse_seed = value

    def build_ends(self):
        delattr(self, 'layers')

        self.ends = nn.ModuleDict({
            lang: RevTransformerLangEnd(self.args) for lang in self.langs
        })
        if self.args.share_all:
            for key in self.ends:
                self.ends[key] = self.ends[self.langs[0]]

        if self.args.lang_end_inner_order == 'fs':
            for key in self.ends:
                self.ends[key].forward, self.ends[key].reverse = self.ends[key].reverse, self.ends[key].forward

    def build_out_norm(self):
        feature_dim = self.args.encoder_embed_dim * 2

        if self.args.out_norm_type == 'actnorm':
            _actnorm = ActNorm(feature_dim)
            self.actnorms = nn.ModuleDict({
                lang: _actnorm for lang in self.langs
            })

    def in_norm(self, x, mask, lang):
        if not self.args.split_embedding:
            # concat same embedding to match feature dimension of revnet.
            x = torch.cat([x, x], dim=-1)

        if self.args.out_norm_type == 'actnorm':
            o = self.actnorms[lang](x, mask)
            return o
        return x

    def out_norm(self, x, mask, lang):
        o = x
        if self.args.out_norm_type == 'actnorm':
            o = self.actnorms[lang].reverse(o, mask)

        if not self.args.split_embedding:
            # split feature to match embedding dimension
            o_1, o_2 = torch.chunk(o, 2, -1)
            o = (o_1 + o_2) / 2

        return o

    def forward(self, x, src_emb, src_padding_mask, src_lang='de', tgt_lang='en', return_all_hiddens=False):
        src_end = self.ends[src_lang]
        tgt_end = self.ends[tgt_lang]

        # 1) x_emb -> source end
        x = self.in_norm(x, ~src_padding_mask, lang=src_lang)

        # 2) encoding: run source end to get intermidiate representation
        mid, src_end_states = src_end.forward(x, src_padding_mask)

        # 3) decoding: run target end to get output feature 
        out, tgt_end_states = tgt_end.reverse(mid, src_padding_mask)

        # 4) target end -> y_emb
        y = self.out_norm(out, ~src_padding_mask, lang=tgt_lang)

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)
            encoder_states.extend(src_end_states)
            encoder_states.extend(tgt_end_states)

        return EncoderOut(
            encoder_out=y,  # B x T x C
            encoder_mid=mid,  # B x T x C
            encoder_padding_mask=src_padding_mask,  # B x T
            encoder_embedding=src_emb,  # B x T x C
            encoder_states=encoder_states,  # List[B x T x C]
            src_tokens=None,
            src_lengths=None,
        )

    def reverse(self, **kwargs):
        src_lang, tgt_lang = kwargs.pop('src_lang'), kwargs.pop('tgt_lang')
        return self.forward(src_lang=tgt_lang, tgt_lang=src_lang, **kwargs)


class SimpleDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(TransformerDecoder, self).__init__(dictionary)

        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = args.encoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = self.embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

    def forward(self, normalize, encoder_out, prev_output_tokens=None, step=0, **unused):
        features = encoder_out.encoder_out
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    # args.encoder_normalize_before = getattr(
    #     args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    # REDER specific 
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.self_attention_type = getattr(args, "self_attention_type", "relative")
    args.upsampling = getattr(args, "upsampling", 2)
    args.ctc_loss = getattr(args, 'ctc_loss', True)



@register_model_architecture("mREDER", "mREDER_wmt_en_de")
def REDER_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("mREDER", "mREDER_wmt_en_de_big")
def REDER_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)

    base_architecture(args)


@register_model_architecture("mREDER", "mREDER_iwslt_de_en")
def rev_nat_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.3)

    base_architecture(args)
