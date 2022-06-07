cd $(dirname $0)
code='..'
data="/root/research/data/iwslt14.tokenized.de-en"/binarized
src=en
tgt=de


mkdir -p results

export CUDA_VISIBLE_DEVICES=0

fairseq-generate ${data} --fp16 \
 --gen-subset test -s $src -t $tgt \
 --path checkpoints/checkpoint_best.pt \
 --batch-size 1024 --max-tokens 8192 --beam 4 --remove-bpe \
 > results/test.gen

fairseq-generate ${data} --fp16 \
 --gen-subset train -s $src -t $tgt \
 --path checkpoints/checkpoint_best.pt \
 --batch-size 1024 --max-tokens 8192 --beam 4 --remove-bpe \
 > results/train.kd.gen

grep ^S results/train.kd.gen | cut -f2- > train.$src
grep ^H results/train.kd.gen | cut -f3- > train.kd.$tgt
# 
distil_data=iwslt14.tokenized.distil.de-en
mkdir -p ${distil_data}
# 
mv train.$src train.kd.$tgt $distil_data
# 
cp $data/../code $data/../valid.* $data/../test.* ${distil_data}

cd $distil_data
subword-nmt apply-bpe -c code < train.kd.$src > train.$src && rm train.kd.$src
subword-nmt apply-bpe -c code < train.kd.$tgt > train.$tgt && rm train.kd.$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref train --validpref valid --testpref test \
    --destdir binarized \
    --workers 20 \
    --srcdict $data/dict.$src.txt \
    --tgtdict $data/dict.$tgt.txt 
