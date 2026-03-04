'''
Why this code exists:

1. There is a bug in the code of extract_lvc_fvc_sentences.py that sometimes includes sentences that do NOT contain the light verb construction or full verb construction into the results. This script removes sentences that do not contain those tokens.

2. The extract_lvc_fvc_sentences.py script was executed in multiple runs on an HPC infrastructure to not claim too much computing time in one go. I once forgot to add the file paths of the processed files to the list of files to ignore, which means some sentences are added multiple times. This script removed the duplicate sentences for each verb file.


'''
import os,glob
input_files = [f for f in glob.glob(os.path.join("PATH-TO/extracted_sentences_in_parts/EXTRACTED_LVC_FVC_SENTENCES/COMBINED_lvc_fvc_sentences-_leading_newline_duplicate_sentences", '*.txt'))]

output_dir = "PATH-TO/extracted_sentences_in_parts/EXTRACTED_LVC_FVC_SENTENCES/lvc_fvc_sentences"

def split_at_second_underscore(s):
    parts = s.split("_")
    return "_".join(parts[:2]), "_".join(parts[2:])

for path in input_files:
    with open(path, "r") as f:
        filename = os.path.split(path)[1]
        verb = filename.split(".txt")[0]
        text = f.read().strip().splitlines()

        no_nl_no_dupl = list(set(text))
        if "SUBST" in filename:
            lv,nominal = split_at_second_underscore(verb)
            contains_verb = list(filter(lambda s: lv in s and nominal in s,no_nl_no_dupl))
        else:
            contains_verb = list(filter(lambda s: verb in s,no_nl_no_dupl))
        output_path = os.path.join(output_dir,filename)
        with open(output_path,'w') as out:
            out.writelines(s + '\n' for s in contains_verb)

        