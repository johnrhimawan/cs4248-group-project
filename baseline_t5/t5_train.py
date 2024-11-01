import os
import re
import torch
import spacy
import string
import subprocess

import numpy as np

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from utils import f1_score, m2_formatter, bleu_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

print('load model...')
name = 'google/flan-t5-xl'
# name = 'google-t5/t5-3b'
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

def preprocess(data):
    # punct = re.compile(' ([' + string.punctuation + ']|n\'t)')
    # src = tokenizer('Please correct the grammar: ' + punct.sub(r'\1', data['src']))
    # tgt = tokenizer(punct.sub(r'\1', data['tgt']))

    src = tokenizer('Please correct the grammar: ' + data['src'])
    tgt = tokenizer(data['tgt'])

    return {
        'input_ids': np.array(src['input_ids']),
        'labels': np.array(tgt['input_ids']),
    }

def compute_metrics(pred):
    m2 = m2_formatter()
    nlp = spacy.load('en_core_web_sm')
    inputs = pred.inputs
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    labels = []
    preds = []
    bleus = []
    prompt = 'Please correct the grammar: '
    # indexes = [random.choice(list(range(len(inputs)))) for _ in range(3)]
    indexes = list(range(3))
    for index, (input, label, pred) in enumerate(zip(inputs, label_ids, pred_ids)):
        input = tokenizer.decode([i for i in input if i != -100], skip_special_tokens=True).replace(prompt, '')
        label = tokenizer.decode([i for i in label if i != -100], skip_special_tokens=True)
        pred = tokenizer.decode([i for i in pred if i != -100], skip_special_tokens=True)

        input = ' '.join([i.text for i in nlp(input)])
        label = ' '.join([i.text for i in nlp(label)])
        pred = ' '.join([i.text for i in nlp(pred)])

        if index in indexes:
            print('Source:', input)
            print('Target:', label)
            print('Predic:', pred)
            print()

        labels.append(m2(input, label))
        preds.append(m2(input, pred))
        bleus.append(bleu_score(pred, label))

    return {
        'f1': f1_score(preds, labels),
        'bleu': sum(bleus) / len(bleus),
    }

TRAIN_DIR = '../data/train/'
VAL_DIR = '../data/development/'

print('load dataset...')
train_ds = read_parallel(TRAIN_DIR).shuffle(42).map(preprocess)
train_ds.set_format(type='torch')
val_ds = read_parallel(VAL_DIR).map(preprocess)
val_ds.set_format(type='torch')

print('start training...')
output_dir = '/home/a/adrian/cs4248/flan-t5-xl-punct-grammar-error-correction'
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy='steps',
    # prediction_loss_only=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_strategy='steps',
    logging_first_step=True,
    logging_steps=30000,
    save_strategy='steps',
    save_steps=30000,
    save_total_limit=1,
    bf16=True,
    predict_with_generate=True,
    report_to='none',
    load_best_model_at_end=True,
    include_inputs_for_metrics=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
