# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.models.nat import NATransformerModel, base_architecture
from fairseq.models import register_model, register_model_architecture
from nonauto.criterions.nat_loss import sequence_ctc_loss_with_logits
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut




@register_model("ctc_nat")
class CTCNATModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.blank_index = getattr(decoder.dictionary, "blank_index", None)

    @property
    def allow_ensemble(self):
        return False

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--ctc-loss",
            action="store_true",
            default=False,
            help="use custom param initialization for BERT",
        )    
        parser.add_argument(
            "--upsampling",
            type=int, metavar='N',
            default=1,
            help="upsampling ratio",
        )  
        parser.add_argument(
            "--upsampling-source",
            action="store_true",
            default=False,
            help="upsampling ratio",
        )  

    def _full_mask(self, target_tokens):
        pad = self.pad
        bos = self.bos
        eos = self.eos
        unk = self.unk

        target_mask = target_tokens.eq(bos) | target_tokens.eq(
            eos) | target_tokens.eq(pad)
        return target_tokens.masked_fill(~target_mask, unk)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if self.args.upsampling_source:
            prev_output_tokens = self._full_mask(src_tokens)
        prev_output_tokens = self._maybe_upsample(prev_output_tokens)

        # decoding
        logits = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)
        logit_mask, target_mask = prev_output_tokens.ne(self.pad), tgt_tokens.ne(self.pad)

        # loss
        ctc_loss = sequence_ctc_loss_with_logits(
            logits=logits,
            logit_mask=logit_mask,
            targets=tgt_tokens,
            target_mask=target_mask,
            blank_index=self.blank_index,
            label_smoothing=self.args.label_smoothing,
        )

        net_output = {
            "word_pred_ctc": {
                "loss": ctc_loss,
                "nll_loss": True,
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }

        return net_output

    def _maybe_upsample(self, tokens):
        if self.args.upsampling <= 1:
            return tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x
        return _us(tokens, self.args.upsampling)

    def initialize_output_tokens(self, encoder_out, src_tokens, **kwargs):
        if self.args.upsampling_source:
            initial_output_tokens = self._full_mask(src_tokens)
        else:
            # length prediction
            length_tgt = self.decoder.forward_length_prediction(
                self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out
            )

            max_length = length_tgt.clamp_(min=2).max()
            idx_length = utils.new_arange(src_tokens, max_length)

            initial_output_tokens = src_tokens.new_zeros(
                src_tokens.size(0), max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        # upsampling decoder input here
        initial_output_tokens = self._maybe_upsample(initial_output_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


@register_model_architecture("ctc_nat", "ctc_nat_wmt_en_de")
def ctc_nat_base_architecture(args):
    args.ctc_loss = getattr(args, "ctc_loss", True)
    args.upsampling = getattr(args, "upsampling", 2)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    base_architecture(args)
