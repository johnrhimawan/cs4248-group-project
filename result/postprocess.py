import re

INPUT_FILE = "BEA2019_ENSEMBLE.txt"
OUTPUT_FILE = "BEA2019_ENSEMBLE_postprocessed.txt"

def remove_redundant_question_mark(line):
    return re.sub(r"\( \? \)\n$", "\n", line)

with open(INPUT_FILE, "r") as input_file:
    source = input_file.readlines()

def process_line(line):
    line = remove_redundant_question_mark(line)
    return line

target = map(process_line, source)

with open(OUTPUT_FILE, "w") as output_file:
    output_file.writelines(target)
