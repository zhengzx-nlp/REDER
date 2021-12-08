#! /bin/bash
code="`pwd`/.."

data_bin=/path/to/data-bin/iwslt14.tokenized.distil.de-en
lang_pairs=de-en,en-de
arch="REDER_iwslt_de_en"

extra_args=(
    --upsampling 2 
    --self-attention-type relative
    --out-norm-type actnorm 
    --feature-shuffle 
    --max-tokens 8192  --update-freq 2 
)

echo ${extra_args[@]}

export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ $debug == "debug" ]] ; then
    cmd="python -m debugpy --listen 0.0.0.0:5678 --wait-for-client"
else
    cmd="python"
fi

python ${code}/train.py --fp16 \
    ${data_bin} \
    --user-dir ${code}/nonauto \
    --save-dir ./checkpoints \
    --ddp-backend=no_c10d \
    --arch ${arch} \
    --task translation_nat_multi 
    --criterion my_nat_loss \
    --lang-pairs ${lang_pairs} \
    --noise no_noise \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 4000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --encoder-learned-pos \
    --apply-bert-init \
    --fixed-validation-seed 7 \
    --keep-last-epochs 5 --keep-best-checkpoints 5 \
    --max-update 600000 \
    --log-format simple \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    $(echo ${extra_args[@]}) 


# average checkpoint
python $code/scripts/average_checkpoints \
    --inputs checkpoints \
    --num-best-checkpoints 5 \
    --output checkpoints/checkpoint.avg5.pt