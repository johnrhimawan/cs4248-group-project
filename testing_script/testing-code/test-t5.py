import os
import argparse
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "christopher-ml/flan-t5-xl-grammatical-error-correction"

# Paths for CoNLL-14 and BEA-19 test files
TEST_FILE_CONLL14 = "../data/test/conll14st-test-data/alt/official-2014.combined-withalt.src"
TEST_FILE_BEA19 = "../data/test/ABCN.test.bea19.orig"

OUTPUT_DIR = "result/"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Grammar Correction with T5 model")
    parser.add_argument("--TEST_SET", type=str, choices=["BEA2019", "CONLL14"], required=True,
                        help="Specify the test set to use: BEA2019 or CONLL14")
    return parser.parse_args()

def determine_test_file(test_set):
    if test_set == "CONLL14":
        return TEST_FILE_CONLL14, "t5-conll14"
    else:
        return TEST_FILE_BEA19, "t5-bea19"

def get_unique_output_filename(base_filename):
    index = 0
    while True:
        output_file = os.path.join(OUTPUT_DIR, f"{base_filename}-{index}.txt")
        if not os.path.exists(output_file):
            return output_file
        index += 1

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
    args = parse_arguments()
    test_file, base_filename = determine_test_file(args.TEST_SET)
    output_file = get_unique_output_filename(base_filename)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Loading test data from {test_file}...")
    test_sentences = load_test_data(test_file)
    
    print("Generating corrected sentences...")
    corrected_sentences = correct_sentences(model, tokenizer, test_sentences)
    
    print(f"Saving corrected sentences to {output_file}...")
    save_corrected_sentences(corrected_sentences, output_file)
    
    print("Done! Corrected sentences saved.")

if __name__ == "__main__":
    main()
