cd $(dirname $0)

code='..'
data="/root/research/data/iwslt14.tokenized.de-en/"
src=de
tgt=en


export CUDA_VISIBLE_DEVICES=0
fairseq-train \
	${data}/binarized -s ${src} -t ${tgt} \
    --user-dir ${code}/nonauto \
    --save-dir ./checkpoints \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--fp16 \
| tee log.txt
