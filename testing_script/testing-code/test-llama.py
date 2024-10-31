import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION_PROMPT = "Correct the grammatical errors in the following sentence. Only provide the corrected sentence without any additional explanation or response."
MODEL_ID = "johnrhimawan/Llama-3.1-8B-Instruct-Grammatical-Error-Correction"
TEST_FILE = "../data/test/conll14st-test-data/alt/official-2014.combined-withalt.src" 
OUTPUT_FILE = "result/llama_corrected.txt"  

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bos_token = tokenizer.bos_token or "<|begin_of_text|>"
eos_token = tokenizer.eos_token or "<|end_of_text|>"

def generate_test_prompt(sentence):
    # Sandwich only the sentence with bos_token and eos_token
    return f"{INSTRUCTION_PROMPT}\nOriginal Sentence: {bos_token}{sentence}{eos_token}\nCorrected Sentence:"

def load_test_data(test_file):
    with open(test_file, "r") as f:
        test_sentences = [line.strip() for line in f.readlines()]
    return test_sentences

def correct_sentences(model, tokenizer, sentences):
    corrected_sentences = []
    for sentence in sentences:
        prompt = generate_test_prompt(sentence)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=512)

        corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_sentences.append(corrected_sentence.strip())

        print(f"\nOriginal Input: {sentence}")
        print(f"Input Prompt: {prompt}")
        print(f"Predicted Corrected Sentence: {corrected_sentence}")
    
    return corrected_sentences

def save_corrected_sentences(corrected_sentences, output_file):
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
