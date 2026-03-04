#! /usr/bin/env python
""" Usage: count_sentence_length.py --sentences=<sentences> --output=<output> 
    
Count the lengths of the sentences.
"""


from docopt import docopt
import logging
import os
import glob
import pandas as pd
import re

ARGS = docopt(__doc__)


OUTPUT_PATH = ARGS['--output']
SENTENCES_PATH = ARGS['--sentences']


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def get_summary_stats(numbers):
        stats = pd.Series(numbers).describe()
        stats_list = [stats["count"], stats["min"], stats["25%"], stats["50%"], stats["mean"], stats["std"], stats["75%"], stats["max"]]
        return [str(float(s)) for s in stats_list]


if __name__ == '__main__':

    logging.getLogger("py4j").setLevel(logging.INFO)

    sentences_files = [f for f in glob.glob(os.path.join(SENTENCES_PATH,'*.txt'))]

    length_stats = [["verb", "count", "min", "Q1", "median", "mean", "std", "Q3", "max"]]
    all_lengths = []
    LVC_lengths = []
    FVC_lengths = []
    for sentences_file in sentences_files:
        lengths = []
        with open(sentences_file,'r') as f:
            sentences = f.readlines()
            for s in sentences:
                lengths.append(len(s.split()))
        all_lengths += lengths

        filename = os.path.split(sentences_file)[1]

        if len(re.findall("_", filename)) == 1:
            FVC_lengths += lengths
        else:
            LVC_lengths += lengths

        length_stats += [[filename.split(".")[0]] + get_summary_stats(lengths)]

        output_path = f"{OUTPUT_PATH}/per_verb"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, filename), 'w') as out:
            out.writelines("\n".join([str(c) for c in lengths]))

    meta_overview = [["label", "count", "min", "Q1", "median", "mean", "std", "Q3", "max"],
                     ["combined_overview"] + get_summary_stats(all_lengths),
                     ["LVC_overview"] + get_summary_stats(LVC_lengths),
                     ["FVC_overview"] + get_summary_stats(FVC_lengths)]
        
    with open(f"{OUTPUT_PATH}/meta_overview_sentence_lengths.csv", 'w') as out:
        out.writelines("\n".join([",".join(c) for c in meta_overview]))
                
    
    with open(f"{OUTPUT_PATH}/overview_sentence_lengths.csv", 'w') as out:
        out.writelines("\n".join([",".join(c) for c in length_stats]))
                


    # SENTENCE PROJECTIONS
    # python explore_vectorspace/count_sentence_length.py --sentences=PATH-TO/LVC-event-duration/code/io/lvc_fvc_sentences --output=PATH-TO/LVC-event-duration/code/io/lvc_fvc_sentence_lengths


