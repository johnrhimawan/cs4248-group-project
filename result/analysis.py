def compare_files():
    xl_file = "XL.txt"
    xxl_file = "XXL.txt"
    ensemble_file = "ENSEMBLE.txt"
    output_file = "output.txt"

    with open(xl_file, 'r') as f_xl, open(xxl_file, 'r') as f_xxl, open(ensemble_file, 'r') as f_ensemble:
        xl_lines = f_xl.readlines()
        xxl_lines = f_xxl.readlines()
        ensemble_lines = f_ensemble.readlines()

    if not (len(xl_lines) == len(xxl_lines) == len(ensemble_lines)):
        raise ValueError("Files do not have the same number of lines.")

    xl_only_count = 0
    xxl_only_count = 0
    both_count = 0
    unexpected_count = 0

    results = []
    for xl_sentence, xxl_sentence, ensemble_sentence in zip(xl_lines, xxl_lines, ensemble_lines):
        xl_sentence = xl_sentence.strip()
        xxl_sentence = xxl_sentence.strip()
        ensemble_sentence = ensemble_sentence.strip()
        
        if ensemble_sentence == xl_sentence and ensemble_sentence != xxl_sentence:
            result = f"XL: {xl_sentence}\nXXL: {xxl_sentence}\nXL"
            xl_only_count += 1
        elif ensemble_sentence == xxl_sentence and ensemble_sentence != xl_sentence:
            result = f"XL: {xl_sentence}\nXXL: {xxl_sentence}\nXXL"
            xxl_only_count += 1
        elif ensemble_sentence == xl_sentence == xxl_sentence:
            result = f"XL: {xl_sentence}\nXXL: {xxl_sentence}\nBOTH"
            both_count += 1
        else:
            result = f"XL: {xl_sentence}\nXXL: {xxl_sentence}\nUNEXPECTED: {ensemble_sentence}"
            unexpected_count += 1
        
        results.append(result)

    total_lines = len(ensemble_lines)
    xl_only_percentage = (xl_only_count / total_lines) * 100
    xxl_only_percentage = (xxl_only_count / total_lines) * 100
    both_percentage = (both_count / total_lines) * 100
    unexpected_percentage = (unexpected_count / total_lines) * 100

    with open(output_file, 'w') as f_out:
        f_out.write(f"Summary Table:\n")
        f_out.write(f"XL Only: {xl_only_percentage:.2f}%\n")
        f_out.write(f"XXL Only: {xxl_only_percentage:.2f}%\n")
        f_out.write(f"Both: {both_percentage:.2f}%\n")
        f_out.write(f"Unexpected: {unexpected_percentage:.2f}%\n\n")
        
        for result in results:
            f_out.write(result + "\n\n")

compare_files()
