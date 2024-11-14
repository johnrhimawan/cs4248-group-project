# Evaluating Models

The scripts in this directory are used to evaluate T5 and LLaMA models on the CoNLL-2014 and BEA-2019 test sets for grammatical error correction (GEC).

## Directory Structure

```bash
.
├── eval-conll14.sh
├── testing-code/
│   ├── test-llama.py
│   └── test-t5.py
└── test.sh
```

- `testing-code/test-t5.py`: Script to evaluate the T5 model on a specified test set (CoNLL-2014 or BEA-2019). It loads the model, processes the test data, generates corrected sentences, and saves the output.
- `testing-code/test-llama.py`: Script to evaluate the LLaMA model on the BEA-2019 test set. It constructs prompts, generates corrected sentences, and saves the results.
- `test.sh`: SLURM job script to run `testing-code/test-t5.py` on a computing cluster with GPU support.
- `eval-conll14.sh`: SLURM job script to evaluate the T5 model’s output on the CoNLL-2014 test set using the m2scorer script.

## Modifying Arguments/Variables

To adjust functionality of these script files, modify the following arguments (when running the script) and/or variables (in the script):

`testing-code/test-t5.py`:

- Change Test Set: Modify the `--TEST_SET` argument when running the script.
- Model Selection: Change the `MODEL_ID` variable to switch between different T5 models.
- Output Directory: Modify the `OUTPUT_DIR` variable to change where results are saved.

`testing-code/test-llama.py`:

- Test File: Change the `TEST_FILE` variable to switch between test sets.
- Model Selection: Modify the `MODEL_ID` variable to use a different LLaMA model.
- Output File: Change the `OUTPUT_FILE` variable to specify the result file name.

## Running the Scripts

### Testing T5 Model

To evaluate the T5 model on a specific test set:

```bash
python testing-code/test-t5.py --TEST_SET [TEST_SET_NAME]
```

Replace `[TEST_SET_NAME]` with either CONLL14 or BEA2019. For example:

```bash
python testing-code/test-t5.py --TEST_SET CONLL14
```

### Testing LLaMA Model

To evaluate the LLaMA model:

```bash
python testing-code/test-llama.py
```

The script defaults to using the BEA-2019 test set. Modify the `TEST_FILE` variable in `test-llama.py` to change the test set.

### Evaluating Results

To evaluate the corrected outputs against the gold standard using m2scorer:

```bash
bash eval-conll14.sh
```

Ensure that the paths in `eval-conll14.sh` point to the correct result files and reference data.

## Additional Notes

- Model Weights: Ensure you have access to the specified models (`christopher-ml/flan-t5-xl-grammatical-error-correction`, etc.) via Hugging Face or local files.
- SLURM Scripts: The `#SBATCH` directives in `test.sh` and `eval-conll14.sh` are configured for a specific computing environment. Modify them according to your cluster’s requirements.
- Logging: The scripts print progress messages to the console. Redirect output to a log file if needed.
- Error Handling: The scripts assume that all dependencies and data files are correctly set up. Implement additional error handling as necessary.
