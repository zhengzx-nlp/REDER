#! /bin/bash

cd $(dirname $0)
code=/root/research/projects/REDER
data_bin="/root/research/data/iwslt14.tokenized.distil.de-en"/binarized

lang_pairs="de-en,en-de"
src=en
tgt=de


ckpt=checkpoints/checkpoint_best.pt

mkdir -p results

export CUDA_VISIBLE_DEVICES=0


lp=$src-$tgt
f=results/$lp

at_reranker=/root/research/projects/REDER/experiments/at/$lp/checkpoints/checkpoint_best.pt

function run() {
    local dataset=$1
    local ckpt=./checkpoints/checkpoint_$2.pt

    beam=100  # for ctc beam search
    args=(
        ${data_bin} --gen-subset $dataset --fp16 
        --user-dir "${code}/nonauto" 
        --path ${ckpt} 
        # --ctc-decode-with-beam $beam         # CTC beam search
        # --path ${ckpt}:${at_reranker}        # AT reranking
        # --iter-decode-with-external-reranker # AT reranking
        --task translation_nat_multi 
        --lang-pairs $lang_pairs 
        -s $src -t $tgt 
        --max-len-a 2 --max-len-b 10 --remove-bpe
        --batch-size 1024 --max-tokens 1024
    )


    echo "[$lp] generating $2 ckpt on $1 (b=${beam}, reranking)..."
    python ${code}/fairseq_cli/generate.py "${args[@]}" > $f.gen

    tail -n 1 $f.gen
    bash $code/scripts/compound_split_bleu.sh $f.gen
}

for dataset in valid test ; do 
    for ckpt in best avg5 ; do 
        run $dataset $ckpt
    done
done
