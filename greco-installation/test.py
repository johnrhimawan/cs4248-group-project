import torch
from models import GRECO

print("Packages imported")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GRECO('microsoft/deberta-v3-large').to(device)
model.load_state_dict(torch.load('models/checkpoint.bin'))

print("Model loaded")

source_file = "../cs4248-group-project/data/test/ABCN.test.bea19.orig"
hypotheses_file = "../cs4248-group-project/testing_script/result/t5-bea19-1.txt"
output_file = "testing/output.txt"

with open(source_file, "r") as sf:
    source = sf.readlines()

with open(hypotheses_file) as hf:
    hypotheses = hf.readlines()

print("Files read")

result = model.score(source, hypotheses)

print("Scoring done")

with open(output_file, "w") as of:
    output_file.write(str(result))

print("Everything done!")

