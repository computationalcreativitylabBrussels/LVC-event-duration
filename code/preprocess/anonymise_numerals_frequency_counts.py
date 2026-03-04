#! /usr/bin/env python
""" Usage: anonymise_numerals_frequency_counts.py --freqs=<freqs> --output=<output>
    
Anonymise and accumulate (per part-of-speech) numerals in frequency counts.
"""
from fileinput import filename
from docopt import docopt

import stanza
import pandas as pd
import os
from functools import reduce
import re

ARGS = docopt(__doc__)
FREQ_PATH = ARGS['--freqs']

OUTPUT_PATH = ARGS['--output']
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# https://github.com/xinxinlaoshi/QuickKisses/blob/main/target_event_extraction.py
en_nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)

def add_dictionaries(dict1,dict2):
    for k,v in dict2.items():
        dict1[k] += v
    return dict1

def reduce_list_dictionaries(dicts):
    return reduce(add_dictionaries, dicts[1:], dicts[0])

def anonymise_numerals(df):
    anonymised_numerals = {}

    for row in df.itertuples():
        if row.lemma_POS == "one_PRON":
            continue

        split_lemma_pos = row.lemma_POS.strip(" _").split("_")
        # deal with edge case where there's no lemma but just _POS
        if len(split_lemma_pos) != 2:
            continue
        lemma, pos = split_lemma_pos

        parsed_word = en_nlp_pipeline(lemma).sentences[0].words[0]

        if parsed_word.upos == "NUM" or re.search("\d+", lemma):
            anonymised_numerals[row.id]= (row.id, row.count, f"--anonnumeral_{pos}", f"{parsed_word.lemma}_{pos}")

    anon_num_df = pd.DataFrame(anonymised_numerals.values(),
                       columns=["id","count","lemma_POS","old_lemma_POS"])\
            .drop(["id","old_lemma_POS"], axis="columns")\
            .groupby("lemma_POS").sum().reset_index()
    
    return anonymised_numerals,anon_num_df


if __name__ == "__main__":
    freqs = pd.read_csv(FREQ_PATH,sep="|",names=["id","count","lemma_POS"])
    # print(freqs)

    anonymised_numerals, anon_num_df = anonymise_numerals(freqs)
    # print(anonymised_numerals)
    not_numerals = freqs[~freqs["id"].isin(anonymised_numerals.keys())].drop("id", axis="columns")
    # print(not_numerals)

    final_df = pd.concat([not_numerals, anon_num_df], ignore_index=True)
    # final_df["count"] = pd.to_numeric(final_df["count"], downcast="integer", errors="coerce")
    final_df = final_df.sort_values("count", ascending=False, ignore_index=True)

    final_df["id"] = final_df.index + 1
    final_df = final_df[["id","count","lemma_POS"]]
    

    filename = os.path.split(FREQ_PATH)[1]

    final_df.to_csv(OUTPUT_PATH + "ANON_NUM_-" + filename, sep="|", index=False, header=False)

    # python preprocess/anonymise_numerals_frequency_counts.py --freqs=PATH-TO/LVC-event-duration/data/BNC_COMBINED-freq-no_ART-PUNC-UNC-TRUNC-INTERJ_pipeline.txt --output=PATH-TO/anon_frequencies

