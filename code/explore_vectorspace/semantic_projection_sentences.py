#! /usr/bin/env python
""" Usage: semantic_projection.py --vectorspace=<vectorspace> --basis-dimension=<basis-dimension> --projection-words=<projection-words> --sentences=<sentences> --output=<output> [--dimensions=<dimensions>]
    
Perform semantic projection technique of Grand et al. (2022) on sentences containing the target LVCs or FVCs.
"""
from collections import defaultdict
from functools import reduce
import token
from docopt import docopt
from sympy import false, li

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

VECTORSPACE_PATH = ARGS['--vectorspace']
BASIS_DIMENSION = int(ARGS['--basis-dimension'])
PROJECTION_WORDS_PATH = ARGS['--projection-words']
OUTPUT_PATH = ARGS['--output']
SENTENCES_PATH = ARGS['--sentences']
DIMENSIONS = ARGS['--dimensions']

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def get_unique_tokens(path, header=True):
    with open(path,'r') as file:
        lines = [l.strip('\n') for l in file.readlines()]
        if header:
            lines = lines[1:]
        tokens = re.split(r'\s|,',' '.join(lines))
        return set(tokens)
    
def flatten(l):
    return list(reduce(operator.concat, l))
    
def get_unique_tokens_from_dict(d):
    values  = flatten(list(d.values()))
    split_values = set(flatten(list(map(lambda s: s.split(), values))))
    if type(list(d.keys())[0]) == tuple:
        keys = set(flatten(list(d.keys())))
    else:
        keys = set(list(d.keys()))
    return  keys | split_values
               


    
'''
calculate_scale_vector calculates the scale vector to project on by computing the mean of the two extremes (indicated by the category labels in the CSV file) and subtracting those two vectors from each other
'''
def calculate_scale_vector(df: pd.DataFrame, settings_path = None) -> np.float64:
    def calculate_mean(column) -> np.float64:
        values = [v.values[0] for v in column]
        sum = csr_array(values[0])

        for v in values[1:]:
            sum += csr_array(v)

        return sum/len(values)

    category_1 = df[df['category']==1]['ppmi_sparse']
    category_2 = df[df['category']==2]['ppmi_sparse']

    mean_category_1 = calculate_mean(category_1)
    mean_category_2 = calculate_mean(category_2)

    print("CAT1 NORM: {} CAT2 NORM: {}".format(linalg.norm(mean_category_1),linalg.norm(mean_category_2)))

    if settings_path:
        with open(settings_path, "a") as f:
            line = "\nCAT1 NORM: {} CAT2 NORM: {}".format(linalg.norm(mean_category_1),linalg.norm(mean_category_2))
            f.write(line)


    return  mean_category_2 - mean_category_1

def scalar_projection(target: sparray, scale: sparray, norm: np.float64) -> float:
    projection = target.dot(scale.transpose())/norm
    return float(projection.todense()[0,0]) # output one singular float representing the scalar projection

'''
get_ppmi_sparse obtains the scipy.sparse array representing the PPMI of the given lemma_pos
'''
def get_ppmi_sparse(df,lemma_pos) -> sparray:
    return df[df['lemma_pos']== lemma_pos]['ppmi_sparse'].values[0]

def compose_sentence_vector(sentence_string, vector_space):
    words = sentence_string.split()
    vectors = vector_space[vector_space['lemma_pos'].isin(words)]
    result = lil_array((1,BASIS_DIMENSION))
    for word in words:
        word_ppmi = vectors[vectors['lemma_pos']==word]['ppmi_sparse']
        if word_ppmi.shape[0] == 0:
            return 
        else:
            result += word_ppmi.iloc[0]

    return result
        
def write_results_to_file(results):
    for verb, projections in results.items():
        projections_string = "\n".join(map(str, projections))
        if type(verb) == tuple:
            verb = verb[0] + "_" + verb[1]

        path = OUTPUT_PATH + "/"+ verb + "_projections.txt"
        write_to_file(path, projections_string)

if __name__ == '__main__':

    logging.getLogger("py4j").setLevel(logging.INFO)

    settings_string = ["{} {}\n".format(e[0], e[1]) for e in ARGS.items()]
    settings_path = os.path.split(OUTPUT_PATH)[0] + "/semantic_projection_settings.txt"
    write_to_file(settings_path,"".join(settings_string))

    projection_words = pd.read_csv(PROJECTION_WORDS_PATH,header=0)
    tokens = set(projection_words.get('lemma_pos').to_list())

    sentences_files = [f for f in glob.glob(os.path.join(SENTENCES_PATH,'*.txt'))]

    for sentences_file in sentences_files:
        with open(sentences_file,'r') as f:
            sentences = f.readlines()
            for s in sentences:
                unique_words = list(set(s.split()))
                tokens = tokens.union(unique_words)

    spark = SparkSession.builder\
                        .config("spark.driver.maxResultSize", "0")\
                        .getOrCreate()

    vector_space = read_ppmi_vector_space(spark, VECTORSPACE_PATH)
    vectors = vector_space.where(vector_space.lemma_pos.isin(tokens)).toPandas()

    selected_dimensions = []
    if DIMENSIONS:
        with open(DIMENSIONS, 'r') as f:
            selected_dimensions = [int(x) for x in f.readline().split(",")]

    # dict of the following form: {lemma_pos: PPMI}
    ppmi_vectors = {}
    for idx,row in vectors.iterrows():
        sparse = convert_to_sparse_vector(BASIS_DIMENSION,row['ppmi'])
        ppmi_vectors[row['lemma_pos']] = sparse if not DIMENSIONS else sparse[:,selected_dimensions]
    # replace the [[index_i,ppmi_i]] column with the sparse PPMI vector
    vectors['ppmi_sparse'] = vectors['lemma_pos'].map(ppmi_vectors)
    vectors = vectors.drop(columns=['ppmi'])

    # obtain a pandas df with lemma_pos,ppmi_sparse,category
    projection_word_vectors = {}
    for idx,row in projection_words.iterrows():
        lemma = row['lemma_pos']
        projection_word_vectors[lemma] = vectors[vectors['lemma_pos']==lemma]['ppmi_sparse']
    projection_words['ppmi_sparse'] = projection_words['lemma_pos'].map(projection_word_vectors)

    scale_vector = calculate_scale_vector(projection_words,settings_path)
    scale_vector_norm = linalg.norm(scale_vector)

    for sentences_file in sentences_files:
        projection_scores = []
        with open(sentences_file,'r') as f:
            sentences = f.readlines()
            for sentence in sentences:
                composed_vector = compose_sentence_vector(sentence,vectors)
                if type(composed_vector) == lil_array:
                    projection_score = scalar_projection(composed_vector, scale_vector, scale_vector_norm)
                    projection_scores.append(projection_score)
        filename = os.path.split(sentences_file)[1]
        output_path = os.path.join(OUTPUT_PATH, filename)
        with open(output_path, 'w') as out:
            for score in projection_scores:
                out.write(f"{score}\n")
            
    # python explore_vectorspace/semantic_projection_sentences.py --vectorspace=PATH-TO/io/ppmi.txt --basis-dimension=2000 --projection-words=PATH-TO/io/duration_liu_chersoni.csv  --sentences=PATH-TO/io/lvc_fvc_sentences --output=PATH-TO/io/lvc_fvc_sentence_projections







