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
fairseq-preprocess --source-lang de --target-lang en \
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

 fairseq-generate /path/to/data-bin/iwslt14.tokenized.de-en \
     --gen-subset train \
     --path checkpoints/checkpoint_best.pt \
     --batch-size 512 --max-tokens 4096 --beam 4 --remove-bpe \
     --results-path results/train.kd.en
```

Extract plain texts
```sh
output=results/train.kd.en
grep ^S $output | cut -f2- > train.de
grep ^H $output | cut -f3- > train.kd.en

mkdir /path/to/iwslt.tokenized.distil.de-en
mv train.de train.kd.en /path/to/iwslt.tokenized.distil.de-en
```

Process/binarize data to fairseq format
```sh
ORI_TEXT=iwslt14.tokenized.de-en
DISTIL_TEXT=iwslt14.tokenized.distil.de-en

# apply bpe using original code
mv ORI_TEXT/code ORI_TEXT/train.de ORI_TEXT/valid.* ORI_TEXT/test.* DISTIL_TEXT 
cd DISTIL_TEXT
subword-nmt apply-bpe -c code < train.kd.en > train.en && rm train.kd.en 

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref train --validpref valid --testpref test \
    --destdir /path/to/data-bin/iwslt14.tokenized.distil.de-en \
    --workers 20 --joint-dictionary \
    --srcdict /path/to/data-bin/iwslt14.tokenized.de-en/dict.de.txt \
    --tgtdict /path/to/data-bin/iwslt14.tokenized.de-en/dict.en.txt 

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




## Citation
```
@inproceedings{zheng2021REDER,
  title={Duplex Sequence-to-Sequence Learning for Reversible Machine Translation},
  author={Zheng, Zaixiang and Zhou, Hao and Huang, Shujian and Chen, Jiajun and Xu, Jingjing and Li, Lei},
  booktitle={NeurIPS},
  year={2021}
}

```
