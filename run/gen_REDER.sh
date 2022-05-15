#! /bin/bash

cd $(dirname $0)
code=..
data_bin=/path/to/data-bin/iwslt14.tokenized.distil.de-en
lang_pairs="de-en,en-de"
src=de
tgt=en


ckpt=checkpoints/checkpoint_best.pt

mkdir -p results

export CUDA_VISIBLE_DEVICES=2


lp=$src-$tgt
f=results/$lp

echo "[$lp] generating best ckpt (b=${beam}, reranking)..."
at_reranker=path/to/at_model/checkpoints/checkpoint_best.pt

beam=20  # for ctc beam search
args=(
    ${data_bin} --gen-subset test --fp16 
    --user-dir "${code}/nonauto" 
    --path ${ckpt} 
    # --ctc-decode-with-beam $beam         # CTC beam search
    # --path ${ckpt}:${at_reranker}        # AT reranking
    # --iter-decode-with-external-reranker # AT reranking
    --task translation_nat_multi 
    --lang-pairs $lang_pairs 
    -s $src -t $tgt 
    --max-len-a 2 --max-len-b 10 --remove-bpe
    --batch-size 1024 --max-tokens 512
)


python ${code}/fairseq_cli/generate.py "${args[@]}" > $f.gen

tail -n 1 $f.gen
bash $code/scripts/compound_split_bleu.sh $f.gen