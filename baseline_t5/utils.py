from collections import Counter
from sacrebleu.metrics import BLEU
from errant.commands.compare_m2 import simplify_edits, process_edits, evaluate_edits, merge_dict, computeFScore

import errant
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate F-scores for error detection and/or correction.\n"
            "Flags let you evaluate at different levels of granularity.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [options] -hyp HYP -ref REF")
    # parser.add_argument(
    #     "-hyp",
    #     help="A hypothesis M2 file.",
    #     required=True)
    # parser.add_argument(
    #     "-ref",
    #     help="A reference M2 file.",
    #     required=True)
    parser.add_argument(
        "-b",
        "--beta",
        help="Value of beta in F-score. (default: 0.5)",
        default=0.5,
        type=float)
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print verbose output.",
        action="store_true")
    eval_type = parser.add_mutually_exclusive_group()
    eval_type.add_argument(
        "-dt",
        help="Evaluate Detection in terms of Tokens.",
        action="store_true")
    eval_type.add_argument(
        "-ds",
        help="Evaluate Detection in terms of Spans.",
        action="store_true")
    eval_type.add_argument(
        "-cs",
        help="Evaluate Correction in terms of Spans. (default)",
        action="store_true")
    eval_type.add_argument(
        "-cse",
        help="Evaluate Correction in terms of Spans and Error types.",
        action="store_true")
    parser.add_argument(
        "-single",
        help="Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1",
        action="store_true")
    parser.add_argument(
        "-multi",
        help="Only evaluate multi token edits; i.e. 2+:n or n:2+",
        action="store_true")
    parser.add_argument(
        "-filt",
        help="Do not evaluate the specified error types.",
        nargs="+",
        default=[])
    parser.add_argument(
        "-cat",
        help="Show error category scores.\n"
            "1: Only show operation tier scores; e.g. R.\n"
            "2: Only show main tier scores; e.g. NOUN.\n"
            "3: Show all category scores; e.g. R:NOUN.",
        choices=[1, 2, 3],
        type=int)
    args = parser.parse_args()
    return args

def f1_score(hyp_m2, ref_m2):
    # Parse command line args
    args = parse_args()
    # Open hypothesis and reference m2 files and split into chunks
    # hyp_m2 = open(args.hyp).read().strip().split("\n\n")
    # ref_m2 = open(args.ref).read().strip().split("\n\n")
    # Make sure they have the same number of sentences
    assert len(hyp_m2) == len(ref_m2)

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(sent[0])
        ref_edits = simplify_edits(sent[1])
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, args)
        ref_dict = process_edits(ref_edits, args)
        # original sentence for logging
        original_sentence = sent[0][2:].split("\nA")[0]
        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(
            hyp_dict, ref_dict, best_dict, sent_id, original_sentence, args)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
    # Print results
    # print_results(best_dict, best_cats, args)
    return computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"], args.beta)[-1]

def m2_formatter():
    annotator = errant.load('en')
    def to_m2(src, tgt):
        src = annotator.parse(src)
        tgt = annotator.parse(tgt)
        edits = annotator.annotate(src, tgt)
        edits = [edit.to_m2() for edit in edits]
        return '\n'.join(edits)
    return to_m2

def bleu_score(hyp, ref):
    return BLEU(effective_order=1).sentence_score(hyp, [ref]).score
