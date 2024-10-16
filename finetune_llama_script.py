import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import bitsandbytes as bnb
from huggingface_hub import login
import os

login(token=os.environ["HUGGINGFACE_TOKEN"])

# No wandb evaluation since I was lazy to setup, perhaps next time

INSTRUCTION_PROMPT = "Correct the grammatical errors in the following sentence:"
BASE_MODEL_ID="meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "llama-3.1-fine-tuned-gec"
MODEL_DIR = "Llama-3.1-8B-Instruct-Grammatical-Error-Correction"
TRAIN_FILES = ["A.train.gold.bea19", "B.train.gold.bea19", "C.train.gold.bea19", "fce.dev.gold.bea19", "fce.train.gold.bea19", "lang8.train.auto.bea19", "nucle.train.gold.bea19"]
DEVELOPMENT_FILES = ["ABCN.dev.gold.bea19"]

def prepare_dataset(files):
    src_sentences = []
    tgt_sentences = []
    for file in files:
        src_file = f"/data/train/{file}.src"
        tgt_file = f"/data/train/{file}.tgt"
        with open(src_file, "r") as src_f, open(tgt_file, "r") as tgt_f:
            src_sentences.extend(src_f.readlines())
            tgt_sentences.extend(tgt_f.readlines())
    dataset = Dataset.from_dict({"text": src_sentences, "label": tgt_sentences})
    return dataset

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def generate_prompt(text, label):
    return f"""{INSTRUCTION_PROMPT}
    Original Sentence: {text}
    Corrected Sentence: {label}""".strip()

def generate_test_prompt(text):
    return f"""{INSTRUCTION_PROMPT}
    Original Sentence: {text}
    Corrected Sentence:""".strip()

def predict(test, model, tokenizer):
    model.eval()
    prompts = [generate_test_prompt(sentence) for sentence in test]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    corrected_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return corrected_sentences

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config,
    
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id

train_data = prepare_dataset(TRAIN_FILES)
eval_data = prepare_dataset(DEVELOPMENT_FILES)
modules = find_all_linear_names(model)
print(f"The target modules are: {' '.join(modules)}")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,                    # directory to save and repository id
    num_train_epochs=1,                       # number of training epochs
    per_device_train_batch_size=1,            # batch size per device during training
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    logging_steps=1,                         
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    # report_to="wandb",                  # report metrics to w&b
    eval_strategy="steps",              # save checkpoint every epoch
    eval_steps = 0.2
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_kwargs={
    "add_special_tokens": False,
    "append_concat_token": False,
    }
)

trainer.train()

model.config.use_cache = True

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model_reload = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model_reload, OUTPUT_DIR)
model = model.merge_and_unload()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
