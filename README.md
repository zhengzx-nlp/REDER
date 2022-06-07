# REDER
[NeurIPS 2021] Duplex Sequence-to-Sequence Learning for Reversible Machine Translation


**Update**
- Dec 8, 2021: code cleansing and refactoring. (*not fully tested*)


**TODO**
- fully tested the code, and elaborate README when I am not that busy.



## Requirement
Our model is built on [fairseq](https://github.com/pytorch/fairseq)
```
fairseq==0.9.0
pytorch==1.6.0
imputer-pytorch (https://github.com/rosinality/imputer-pytorch)
ctcdecode (https://github.com/parlance/ctcdecode.git)
```

Install by
```sh
git clone https://github.com/zhengzx-nlp/REDER.git && cd REDER
bash nonauto/run/install.sh
```



## Training

### Data Processing
We follow the standard procedure provided by the scripts in fairseq. Here we use `iwslt14.de-en` as an example. This is the script [prepare-iwslt14.sh](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh).

#### Download and prepare raw data
```sh
# Download and prepare the data
bash prepare-iwslt14.sh 

# Preprocess/binarize the data
TEXT=/path/to/iwslt14.tokenized.de-en
src=de
tgt=en 

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir /path/to/data-bin/iwslt14.tokenized.de-en \
    --workers 20 --joint-dictionary
```

#### Training an AT model
```sh
export CUDA_VISIBLE_DEVICES=0

EXP_NAME="iwslt14.de-en.transformer"
mkdir $EXP_NAME && cd $EXP_NAME

fairseq-train \
    /path/to/data-bin/iwslt14.tokenized.de-en \
    -s $src -t $tgt \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir "logs/$EXP_NAME" 
```

#### Create sequence-level KD data using the AT model
Translate the whole training data and use the translation results as the dataset for training NAT instead of ground-truth target translation.

```sh
 export CUDA_VISIBLE_DEVICES=0

 mkdir -p results

fairseq-generate ${data} --fp16 \
    --gen-subset train -s $src -t $tgt \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 1024 --max-tokens 8192 --beam 4 --remove-bpe \
    > results/train.kd.gen

grep ^S results/train.kd.gen | cut -f2- > train.$src
grep ^H results/train.kd.gen | cut -f3- > train.kd.$tgt
```

Extract plain texts
```sh
output=results/train.kd.gen
grep ^S results/train.kd.gen | cut -f2- > train.$src
grep ^H results/train.kd.gen | cut -f3- > train.kd.$tgt

```

Process/binarize data to fairseq format
```sh
data=/path/to/iwslt14.tokenized.de-en/
distil_data=/path/to/iwslt14.tokenized.distil.de-en

# apply bpe using original code
mkdir -p ${distil_data}
 
mv train.$src train.kd.$tgt $distil_data
 
cp $data/code $data/valid.* $data/test.* ${distil_data}

cd $distil_data
subword-nmt apply-bpe -c code < train.kd.$src > train.$src && rm train.kd.$src
subword-nmt apply-bpe -c code < train.kd.$tgt > train.$tgt && rm train.kd.$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref train --validpref valid --testpref test \
    --destdir binarized \
    --workers 20 \
    --srcdict $data/binarized/dict.$src.txt \
    --tgtdict $data/binarized/dict.$tgt.txt 

```



#### Create bidirectional KD data
Do the same for the reverse direction (en-de) reusing the same vocabulary, and put the binarized files in the same data fold together with de-en. 

After that, we will get a data folder having a structure like this:
```sh
/path/to/data-bin/iwslt14.tokenized.distil.de-en
├── dict.de.txt      
├── dict.en.txt      
├── preprocess.log   
├── train.de-en.de.bin
├── train.de-en.de.idx
├── train.de-en.en.bin
├── train.de-en.en.idx
├── valid.de-en.de.bin
├── valid.de-en.de.idx
├── valid.de-en.en.bin
├── valid.de-en.en.idx
├── test.de-en.de.bin
├── test.de-en.de.idx
├── test.de-en.en.bin
└── test.de-en.en.idx
├── train.en-de.de.bin
├── train.en-de.de.idx
├── train.en-de.en.bin
├── train.en-de.en.idx
├── valid.en-de.de.bin
├── valid.en-de.de.idx
├── valid.en-de.en.bin
├── valid.en-de.en.idx
├── test.en-de.de.bin
├── test.en-de.de.idx
├── test.en-de.en.bin
└── test.en-de.en.idx
```


### 2. Training REDER
see `nonauto/run/train_REDER.sh`


## Generation
see `nonauto/run/gen_REDER.sh`


## Example
Please check out `experiments` folder for an excutable complete example on iwslt14 en-de.

## Citation
```
@inproceedings{zheng2021REDER,
  title={Duplex Sequence-to-Sequence Learning for Reversible Machine Translation},
  author={Zheng, Zaixiang and Zhou, Hao and Huang, Shujian and Chen, Jiajun and Xu, Jingjing and Li, Lei},
  booktitle={NeurIPS},
  year={2021}
}

```
