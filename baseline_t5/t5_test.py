import os
import torch
import spacy
import string
import subprocess

import numpy as np

from tqdm.auto import tqdm
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from utils import f1_score, m2_formatter, bleu_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

print('load model...')
name = '/home/a/adrian/cs4248/t5'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(name)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.max_new_tokens = 512

def read_parallel(directory):
    src_files = sorted([f for f in os.listdir(directory) if f.endswith('.src')])
    src_sentences = []
    for f in src_files:
        with open(directory + f, 'r') as file:
            src_sentences.extend([line.strip() for line in file])

    tgt_files = [f.replace('.src', '.tgt') for f in src_files]
    tgt_sentences = []
    for f in tgt_files:
        with open(directory + f, 'r') as file:
            tgt_sentences.extend([line.strip() for line in file])

    return Dataset.from_list([{'src': src, 'tgt': tgt} for src, tgt in zip(src_sentences, tgt_sentences)])

VAL_DIR = '../data/train/'

print('load dataset...')
val_ds = read_parallel(VAL_DIR).take(1000)
prompt = 'Please correct the grammar: '
nlp = spacy.load('en_core_web_sm')
m2 = m2_formatter()
labels = []
preds = []

with open('t5_score_train.out', 'w') as f:
    for row in tqdm(val_ds):
        inp = tokenizer(prompt + row['src'], return_tensors='pt').to(model.device)
        pred = tokenizer.decode([i for i in model.generate(**inp, max_new_tokens=512)[0] if i >= 0], skip_special_tokens=True)
        pred = ' '.join([i.text for i in nlp(pred)])
        print('S:', row['src'], file=f)
        print('T:', row['tgt'], file=f)
        print('H:', pred, file=f)

        labels.append(m2(row['src'], row['tgt']))
        preds.append(m2(row['src'], pred))
        print('F1:', f1_score([preds[-1]], [labels[-1]]), file=f)
        print(file=f)

    print('Total F1:', f1_score(preds, labels), file=f)
