#! /usr/bin/env python
""" Usage: get_corresponding_files.py --lvcs-examples=<lvcs-examples> --sentences=<sentences> --output=<output>
    
Obtain a CSV with pairs of files that correspond to LVC_sentences_path,FVC_sentences_path, for each full verb.

This CSV file can serve as input for the Python script that does the semantic projections.
"""
from collections import defaultdict
from functools import reduce
import token
from docopt import docopt
from sympy import false, li
from torch import full

from pyspark.sql import SparkSession

import logging

import operator
from itertools import chain


from util.read import read_ppmi_vector_space
from util.write import write_to_file
from util.misc import convert_to_sparse_vector

import re
import pandas as pd

import os
import glob

from scipy.sparse import csr_array, linalg, sparray, lil_array
import numpy as np

ARGS = docopt(__doc__)

OUTPUT_PATH = ARGS['--output']
SENTENCES_PATH = ARGS['--sentences']
LVCS_EXAMPLES_PATH = ARGS['--lvcs-examples']

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def list_files_recursive(path):
    files = []
    def get_files(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                get_files(full_path)
            else:
                files.append(full_path)
    get_files(path)
    return files

def get_lemmas(projections_path):
    def split_at_second_underscore(s,p):
        parts = s.split("_")
        return ("_".join(parts[:2]), "_".join(parts[2:]),p)
    verb = lambda filename,path: (filename.split(".txt")[0],path) 
    all_files = list_files_recursive(projections_path)
    print(f"NO FILES: {len(all_files)}")
    filenames = [(os.path.split(p)[1],p) for p in all_files]
    verbs = [verb(fn,p) for fn,p in filenames]
    verbs = [split_at_second_underscore(s,p) for s,p in verbs]

    light_verb_constructions = []
    full_verbs = []
    for verb,nominal,path in verbs:
        if verb == '.DS_Store':
            continue
        if nominal:
            light_verb_constructions.append(((verb,nominal),path))
        else:
            full_verbs.append((verb,path))
    print(light_verb_constructions)
    print(full_verbs)
    return full_verbs, light_verb_constructions


if __name__ == '__main__':

    logging.getLogger("py4j").setLevel(logging.INFO)
  


    lvcs_examples = pd.read_csv(LVCS_EXAMPLES_PATH)
    lv_fv_dict = lvcs_examples.groupby(by=["FVC_POS"])[["LV_POS","nominal_POS"]].apply(lambda x: x.values.tolist()).to_dict()

    path_full_verbs, path_light_verbs = get_lemmas(SENTENCES_PATH)

    matched_files = defaultdict(list)
    
    for fv,path in path_full_verbs:
        lvc = lv_fv_dict[fv]
        in_both = set(map(tuple, lvc)).intersection(set(map(lambda x: tuple(x[0]), path_light_verbs)))
        if in_both:
            matched_files[fv] = list(in_both.union(set(matched_files[fv])))

    matched_paths = []
    path_full_verbs_dict = dict(path_full_verbs)
    path_light_verbs_dict =dict(path_light_verbs)
    for fv,lvs in matched_files.items():
        for lv in lvs:
            matched_paths.append((fv,lv[0],lv[1],path_full_verbs_dict[fv],path_light_verbs_dict[lv]))

    print(matched_paths)

    matched_paths_df = pd.DataFrame(matched_paths, columns =["FV","LV","nominal","FVC","LVC"])

    matched_paths_df.to_csv(OUTPUT_PATH+"/corresponding_filepaths.txt",index=False)
    

# python preprocess/get_corresponding_files.py --lvcs-examples=PATH-TO/LVC-event-duration/data/LVCs/EN_LVCs_single_word_fvc.csv --sentences=PATH-TO/LVC-event-duration/code/io/lvc_fvc_sentences --output=PATH-TO/LVC-event-duration/code/io


