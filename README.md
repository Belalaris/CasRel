# CasRel Model reimplement
The code is the PyTorch reimplement and development of the paper "A Novel Cascade Binary Tagging Framework for Relational Triple Extraction" ACL2020. 
The [official code](https://github.com/weizhepei/CasRel) was written in Keras. 

# Introduction
I followed the previous work of [Onion12138](https://github.com/Onion12138/CasRelPyTorch), [longlongman](https://github.com/longlongman/CasRel-pytorch-reimplement) and [JuliaSun623](https://github.com/JuliaSun623/CasRel_fastNLP), so I have to express sincere thanks to them.

Some changes are made in order to better apply to the Chinese Dataset and put the model into use.
The changes contain:

- I used the tokenizer BertTokenizer, so Chinese sentences are tokenized by single character. 
- I substituted the original pretrained model with 'bert-wwm-chinese'.
- I used fastNLP to build the datasets.
- I changed the encoding and decoding methods in order to fit the Chinese Dataset.
- I reconstructed the structure for readability.
- I added checkpoint saving for epoch with best performance.
- I implemented the Evaluation and Prediction interface for direct use.
# Requirements
- torch==1.8.0+cu111 (In fact there is no need to use such high edition, just choose according to the cudn of your machine)
- transformers==4.3.3
- fastNLP==0.6.0
- tqdm==4.59.0
- numpy==1.20.1
### install
- conda create -n casrel python==3.7
- pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
- pip install transformers==4.3.3
- pip install fastNLP==0.6.0
- pip install tqdm==4.59.0
# Dataset

The training process only use the '**train**' dataset, and evaluation/prediction process use only one dataset that you choose. ('**test**' dataset for the suggested one)

And then 'dev' dataset is for no use since callback does not influence the training process at all

## Baidu

I preprocessed the open-source dataset from Baidu. I did some cleaning, so the data given have 18 relation types. 
Some noisy data are eliminated.

The data are in form of json. Take one as an example:
```json
{
    "text": "陶喆的一首《好好说再见》推荐给大家，希望你们能够喜欢",
    "spo_list": [
        {
            "predicate": "歌手",
            "object_type": "人物",
            "subject_type": "歌曲",
            "object": "陶喆",
            "subject": "好好说再见"
        }
    ]
}
```
In fact the field 'object_type' and 'subject_type' are **not** used.

Some basic statistics:

| dataset | number of sentences | number of triples |
| ------- | ------------------- | ----------------- |
| train   | 55959               | 85131             |
| dev     | 11191               | 17026             |
| test    | 13417               | 24370             |

## DuIE

Baidu dataset is actually a part of DuIE, so if you want to get a better model, choose this larger dataset for training. 
The data structure is the same as that of Baidu.

I also did some statistics:

| dataset | number of sentences | number of triples | max length of sentences |
| ------- | ------------------- | ----------------- | ----------------------- |
| train   | 181787              | 329784            | 300                     |
| test    | 10000               | 18383             | 290                     |


If you have your own data, you can organize your data in the same format.

## encoding
keep encoding='utf-8'  for best


# Usage
```
python Run.py
```
I have already set the default value of the model, but you can still set your own configuration in **model/config.py** and arguments when running **Run.py**

```
python Evaluation.py
```

I have set **five** kinds of computing method for triples(s, r, o), relation, entity(s, o), subject and object, respectively. Choose one or more methods to evaluate a trained model.

```
python Prediction.py
```

Prediction is a basically implemented interface. 
Input a single sentence, and the corresponding output will be in form of json:

```json
{
	"entity": ["ent1", "ent2", "ent3", "ent4"], 
	"relation": [
			["ent1", "ent2"],
			["ent3", "ent4"],
			["ent2", "ent1"]
		],
	"relation_type": ["rel1", "rel2", "rel3"]
}
```

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20210804160443578.png" alt="image-20210804160443578">


# Results

## Baidu

Evaluation results for 15 epochs: 

| evaluation kind | f1     | precision | recall |
| --------------- | ------ | --------- | ------ |
| triples         | 0.7311 | 0.7827    | 0.6858 |
| relation        | 0.8492 | 0.9118    | 0.7945 |
| entity          | 0.7432 | 0.7937    | 0.6987 |
| subject         | 0.8984 | 0.9129    | 0.8843 |
| object          | 0.7933 | 0.8519    | 0.7423 |

## DuIE

| evaluation kind | f1     | precision | recall |
| --------------- | ------ | --------- | ------ |
| triples         | 0.7167 | 0.7253    | 0.7083 |
| relation        | 0.8794 | 0.9022    | 0.8578 |
| entity          | 0.7277 | 0.7359    | 0.7196 |
| subject         | 0.8140 | 0.8390    | 0.7904 |
| object          | 0.8195 | 0.8412    | 0.7989 |

It may not reach its utmost since there is only 30 epochs.

# Experiences
- Learning rate 1e-5 seems a good choice. If you change the learning rate, the model will be dramatically affected.
- It shows little improvement when I substitute BERT with RoBERTa.
- It is crucial to shuffle the datasets in order to avoid overfitting. 
- Do not set the batch size too big/small, 8 is OK. Or the model will be affected to some extent/difficult to converge.
- When training with GPUs, pay attention to the device setting in Run.py, model/data.py, model/evaluate.py.
- Remember to change the name of .pkl of checkpoints and choose the right one for loading.
