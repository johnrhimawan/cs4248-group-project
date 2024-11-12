import os
import torch
from models import GRECO

print("Packages imported")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GRECO('microsoft/deberta-v3-large').to(device)
model.load_state_dict(torch.load('models/checkpoint.bin'))

print("Model loaded")

source_file = "../cs4248-group-project/data/test/ABCN.test.bea19.orig"
hypotheses_file_llama = "corrected_sentences_llama_bea19.txt"
hypotheses_file_t5 = "t5-bea19-3.txt"

output_dir = "ensemble-output"
output_file = f"{output_dir}/ensemble-bea19.txt"
output_file_llama = f"{output_dir}/score-bea19-llama.txt"
output_file_t5 = f"{output_dir}/score-bea19-t5.txt"

with open(source_file, "r") as sf:
    source = sf.readlines()

with open(hypotheses_file_llama) as hf_llama:
    hypotheses_llama = hf_llama.readlines()

with open(hypotheses_file_t5) as hf_t5:
    hypotheses_t5 = hf_t5.readlines()

assert len(source) == len(hypotheses_llama)
assert len(hypotheses_llama) == len(hypotheses_t5)

print("Files read")

result = []
result_llama = []
result_t5 = []
lower_bound = 0
i_iter = 0
while i_iter < 100 and lower_bound < len(source):
    upper_bound = min(lower_bound + 100, len(source))
    
    segment_source = source[lower_bound:upper_bound]
    segment_llama = hypotheses_llama[lower_bound:upper_bound]
    segment_t5 = hypotheses_t5[lower_bound:upper_bound]

    score_llama = model.score(segment_source, segment_llama).cpu()
    score_t5 = model.score(segment_source, segment_t5).cpu()
    comparision = score_t5 > score_llama
    result.extend(sentence_t5 if cmp_res else sentence_llama
                  for cmp_res, sentence_llama, sentence_t5
                  in zip(comparision, segment_llama, segment_t5))
    result_llama.extend(score_llama)
    result_t5.extend(score_t5)
    lower_bound = upper_bound
    i_iter += 1

print("Evaluating done")

os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w") as of:
    of.writelines(result)

with open(output_file_llama, "w") as of_llama:
    of_llama.write(str(result_llama))

with open(output_file_t5, "w") as of_t5:
    of_t5.write(str(result_t5))

print("Everything done!")
