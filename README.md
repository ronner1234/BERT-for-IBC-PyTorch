# How to Fine-Tune PyTorch BERT for Classification of Political Speeches?

This repository contains an op-for-op PyTorch reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation is provided with [Google's pre-trained models](https://github.com/google-research/bert), examples, notebooks and a command-line interface to load any pre-trained TensorFlow checkpoint for BERT is also provided.

As a basis for our investigation we have used [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583). It includes several experiments to investigate different fine-tuning methods of BERT on text classification task were conducted and provide a general solution for BERT fine-tuning.

## Installation

This repo was tested on Python 3.7 and [PyTorch 1.10.1 + cu11.3](https://pytorch.org/)

### With pip

PyTorch pretrained bert can be installed by pip as follows:
```bash
pip install pytorch-pretrained-bert
```

###Additional Requirements
+ spacy
+ pandas
+ numpy

## Run the code

### 1) Prepare the data set:

#### Ideological Books Corpus

The source of the data set: [Ideological Books Corpus](https://people.cs.umass.edu/~miyyer/ibc/index.html)

"The Ideological Books Corpus (IBC) consists of 4,062 sentences annotated for political ideology at a sub-sentential 
level as described in our paper. Specifically, it contains 2025 liberal sentences, 1701 conservative sentences, 
and 600 neutral sentences. Each sentence is represented by a parse tree where annotated nodes are associated with a 
label in {liberal, conservative, neutral}." 

To obtain the full dataset, or for any questions / comments about the data, send an email at miyyer@umd.edu.

#### Prepare the Data for the Model

We devided the Ideological Books Corpus into two csv files. One for training the model, the other for testing the model:

```shell
1: train.csv  (80% of data)
2: test.csv   (20% of data)
```

### Loading a pretrained BERT-Model

When running code it download the pre-trained weights from AWS S3 (see the links [here](pytorch_pretrained_bert/modeling.py)) and store them in a cache folder to avoid future download (the cache folder can be found at `~/.pytorch_pretrained_bert/`).

We will be using the BERT-Uncased model for our pretraining as the input text uses casing very inconsistently and therefore want to neglect it. 

`Uncased` means that the text has been lowercased before WordPiece tokenization, e.g., `John Smith` becomes `john smith`. The Uncased model also strips out any accent markers. `Cased` means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging). For information about the Multilingual and Chinese model, see the [Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md) or the original TensorFlow repository.

**When using an `uncased model`, make sure to pass `--do_lower_case` to the example training scripts (or pass `do_lower_case=True` to FullTokenizer if you're using your own script and loading the tokenizer your-self.).**

The initial download will take a couple of minutes (approx 400mb)

### Pre-Training

Run pretraining:

Number of training epochs (steps): 750 (~ 100.000)

```shell
python run_lm_finetuning.py --bert_model bert-base-uncased --do_train --do_lower_case --train_file ../data/political_corpus.txt --output_dir models --num_train_epochs 750.0 --learning_rate 1e-5 --train_batch_size 32 --max_seq_length 128
```

### Classifying

Number of training epochs: 8

```shell
python run_classifier.py --task_name political --do_train --do_eval --do_lower_case --data_dir ../data/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 8.0 --output_dir ../output/
```