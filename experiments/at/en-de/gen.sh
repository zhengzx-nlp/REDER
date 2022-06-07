cd $(dirname $0)
code=/root/research/projects/REDER/

data="/root/research/data/iwslt14.tokenized.de-en"/binarized
src=en
tgt=de


mkdir -p results

export CUDA_VISIBLE_DEVICES=0

function run() {
	local dataset=$1
	local ckpt=./checkpoints/checkpoint_$2.pt

	f=results/$dataset.$2.gen
	fairseq-generate ${data} --fp16 \
	 --gen-subset ${dataset} -s $src -t $tgt \
	 --path $ckpt \
	 --batch-size 1024 --max-tokens 8192 --beam 4 --remove-bpe \
	 > $f

	tail -n 1 $f
    bash $code/scripts/compound_split_bleu.sh $f

}

# average checkpoint
python $code/scripts/average_checkpoints.py \
    --inputs checkpoints \
    --num-epoch-checkpoints 5 \
    --output checkpoints/checkpoint_avg5.pt

for dataset in valid test ; do
	for ckpt in best avg5 ; do
		run $dataset $ckpt
	done
done
