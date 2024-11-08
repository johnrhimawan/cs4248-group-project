import os
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "christopher-ml/flan-t5-xl-grammatical-error-correction"

# Toggle test file selection
USE_CONLL14 = True  # Set to False to use BEA-19 test file

# Paths for CoNLL-14 and BEA-19 test files
TEST_FILE_CONLL14 = "../data/test/conll14st-test-data/alt/official-2014.combined-withalt.src"
TEST_FILE_BEA19 = "../data/test/ABCN.test.bea19.orig"

TEST_FILE = TEST_FILE_CONLL14 if USE_CONLL14 else TEST_FILE_BEA19

OUTPUT_FILE_CONLL14 = "result/t5-conll14-2.txt"
OUTPUT_FILE_BEA19 = "result/t5-bea19-2.txt"

OUTPUT_FILE = OUTPUT_FILE_CONLL14 if USE_CONLL14 else OUTPUT_FILE_BEA19

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

def load_test_data(test_file):
    with open(test_file, "r") as f:
        test_sentences = [line.strip() for line in f.readlines()]
    return test_sentences

def correct_sentences(model, tokenizer, sentences):
    corrected_sentences = [] 
    nlp = spacy.load('en_core_web_sm')
    for sentence in sentences:
        print(f"Original Sentence: {sentence}")
        
        prompt = f"Please correct the grammar: {sentence}"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=128) 
        
        corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_sentence = ' '.join([w.text for w in nlp(corrected_sentence)])
        corrected_sentences.append(corrected_sentence)

        print(f"Corrected Sentence: {corrected_sentence}\n")
        
    return corrected_sentences

def save_corrected_sentences(corrected_sentences, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for sentence in corrected_sentences:
            f.write(sentence + "\n")

def main(): 
    print(f"Loading test data from {TEST_FILE}...")
    test_sentences = load_test_data(TEST_FILE)
    
    print("Generating corrected sentences...")
    corrected_sentences = correct_sentences(model, tokenizer, test_sentences)
    
    print(f"Saving corrected sentences to {OUTPUT_FILE}...")
    save_corrected_sentences(corrected_sentences, OUTPUT_FILE)
    
    print("Done! Corrected sentences saved.")

if __name__ == "__main__":
    main()
