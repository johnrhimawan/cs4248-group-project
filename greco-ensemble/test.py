import os
import torch
from models import GRECO

print("Packages imported")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GRECO('microsoft/deberta-v3-large').to(device)
model.load_state_dict(torch.load('models/checkpoint.bin'))

print("Model loaded")

source_file = "~/cs4248-group-project/data/test/ABCN.test.bea19.orig"
hypotheses_file_1 = "t5-bea19-XXL.txt"
hypotheses_file_2 = "t5-bea19-XL.txt"

with open(source_file, "r") as sf:
    source = sf.readlines()

with open(hypotheses_file_1) as hf_1:
    hypotheses_1 = hf_1.readlines()

with open(hypotheses_file_2) as hf_2:
    hypotheses_2 = hf_2.readlines()

assert len(source) == len(hypotheses_1)
assert len(source) == len(hypotheses_2)

print("Files read")

result = []
result_1 = []
result_2 = []
lower_bound = 0
i_iter = 0
n_iters = 100
step_size = 100
while i_iter < n_iters and lower_bound < len(source):
    upper_bound = min(lower_bound + step_size, len(source))

    segment_source = source[lower_bound:upper_bound]
    segment_1 = hypotheses_1[lower_bound:upper_bound]
    segment_2 = hypotheses_2[lower_bound:upper_bound]

    score_1 = model.score(segment_source, segment_1).cpu()
    score_2 = model.score(segment_source, segment_2).cpu()
    comparision = score_1 > score_2
    result.extend(sentence_1 if cmp_res else sentence_2
                  for cmp_res, sentence_1, sentence_2
                  in zip(comparision, segment_1, segment_2))
    result_1.extend(score_1.tolist())
    result_2.extend(score_2.tolist())
    lower_bound = upper_bound
    i_iter += 1

print("Evaluating done")

output_dir = "ensemble-output"
output_file = f"{output_dir}/ensemble-bea19.txt"
output_file_1 = f"{output_dir}/score-bea19-t5-XXL.txt"
output_file_2 = f"{output_dir}/score-bea19-t5-XL.txt"

os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w") as of:
    of.writelines(result)

with open(output_file_1, "w") as of_1:
    of_1.writelines('\n'.join(map(str, result_1)))

with open(output_file_2, "w") as of_2:
    of_2.writelines('\n'.join(map(str, result_2)))

print("Everything done!")
