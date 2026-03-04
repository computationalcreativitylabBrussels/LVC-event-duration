#! /usr/bin/env python
""" Usage: semantic_projection.py --vectorspace=<vectorspace> --basis-dimension=<basis-dimension> --lvcs=<lvcs> --projection-words=<projection-words --output=<output>
    
Perform semantic projection technique of Grand et al. (2022) of LVCs and FVCs in isolation (without surrounding context words).
"""
from docopt import docopt

from pyspark.sql import SparkSession

import logging


from util.read import read_ppmi_vector_space
from util.write import write_to_file
from util.misc import convert_to_sparse_vector

import re
import pandas as pd

import os

from scipy.sparse import csr_array, linalg, sparray, issparse
import numpy as np

ARGS = docopt(__doc__)

VECTORSPACE_PATH = ARGS['--vectorspace']
BASIS_DIMENSION = int(ARGS['--basis-dimension'])
PROJECTION_WORDS_PATH = ARGS['--projection-words']
OUTPUT_PATH = ARGS['--output']
LVCS = ARGS['--lvcs']

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def get_unique_tokens(path):
    with open(path,'r') as file:
        lines = [l.strip('\n') for l in file.readlines()][1:]
        tokens = re.split(r'\s|,',' '.join(lines))
        return set(tokens)
    
'''
calculate_scale_vector calculates the scale vector to project on by computing the mean of the two extremes (indicated by the category labels in the CSV file) and subtracting those two vectors from each other
'''
def calculate_scale_vector(df: pd.DataFrame) -> np.float64:
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

    return mean_category_1 - mean_category_2


def scalar_projection(target: sparray, scale: sparray, norm: np.float64) -> float:
    projection = target.dot(scale.transpose())/norm
    return float(projection.todense()[0,0]) # output one singular float representing the scalar projection


'''
get_ppmi_sparse obtains the scipy.sparse array representing the PPMI of the given lemma_pos
'''
def get_ppmi_sparse(df,lemma_pos) -> sparray:
    return df[df['lemma_pos']== lemma_pos]['ppmi_sparse'].values[0]

if __name__ == '__main__':

    logging.getLogger("py4j").setLevel(logging.INFO)

    settings_string = ["{} {}\n".format(e[0], e[1]) for e in ARGS.items()]
    settings_path = OUTPUT_PATH + "/semantic_projection_settings.txt"
    write_to_file(settings_path,"".join(settings_string))

    lvcs = pd.read_csv(LVCS,header=0)
    tokens = get_unique_tokens(LVCS)
    projection_words = pd.read_csv(PROJECTION_WORDS_PATH,header=0)
    projection_word_lemmas = set(projection_words.get('lemma_pos').to_list())
    tokens = tokens.union(projection_word_lemmas)

    spark = SparkSession.builder\
                        .config("spark.driver.maxResultSize", "0")\
                        .getOrCreate()

    vector_space = read_ppmi_vector_space(spark, VECTORSPACE_PATH)
    vectors = vector_space.where(vector_space.lemma_pos.isin(tokens)).toPandas()

    ppmi_vectors = {}
    for idx,row in vectors.iterrows():
        sparse = convert_to_sparse_vector(BASIS_DIMENSION,row['ppmi'])
        ppmi_vectors[row['lemma_pos']] = sparse

    vectors['ppmi_sparse'] = vectors['lemma_pos'].map(ppmi_vectors)
    vectors = vectors.drop(columns=['ppmi'])


    projection_word_vectors = {}
    for idx,row in projection_words.iterrows():
        lemma = row['lemma_pos']
        projection_word_vectors[lemma] = vectors[vectors['lemma_pos']==lemma]['ppmi_sparse']

    projection_words['ppmi_sparse'] = projection_words['lemma_pos'].map(projection_word_vectors)

    scale_vector = calculate_scale_vector(projection_words)
    scale_vector_norm = linalg.norm(scale_vector)

    output_list = [["FVC","FVC_projection","LV","LV_noun","LVC_projection"]]
    not_found_list = []
    for idx,row in lvcs.iterrows():
        full_verb = row['FVC_POS']
        light_verb = row['LV_POS']
        noun = row['nominal_POS']
        vector_space_entries = vectors[vectors['lemma_pos'].isin([full_verb, light_verb, noun])]

        if vector_space_entries.shape[0] != 3:
            not_found = list(set([full_verb, light_verb, noun])-set(vectors['lemma_pos']))
            not_found_list += not_found
            print("The following vectors could be found in the vector space: {}".format(not_found))
            continue

        full_verb_vector = get_ppmi_sparse(vector_space_entries,full_verb)
        light_verb_vector = get_ppmi_sparse(vector_space_entries,light_verb)
        noun_vector = get_ppmi_sparse(vector_space_entries,noun)

        fvc_projection = scalar_projection(full_verb_vector,scale_vector,scale_vector_norm)

        lvc_composition = light_verb_vector + noun_vector
        lvc_projection = scalar_projection(lvc_composition,scale_vector,scale_vector_norm)
        output_list.append([full_verb,str(fvc_projection),light_verb,noun,str(lvc_projection)])

    output_string = "\n".join([",".join(line) for line in output_list])
    not_found_string = ",".join(not_found_list)

    write_to_file(OUTPUT_PATH + "/semantic_projections.txt", output_string)
    write_to_file(OUTPUT_PATH + "/semantic_projections_not_found.txt", not_found_string)
        

        
    
    # Validation of the duration scale
    
    # python explore_vectorspace/semantic_projection.py --vectorspace=PATH-TO/LVC-event-duration/code/io/ppmi.txt --basis-dimension=2000 --lvcs=PATH-TO/LVC-event-duration/code/io/LVC-FVC-pairs.csv --projection-words=PATH-TO/LVC-event-duration/code/io/duration_liu_chersoni.csv --output=PATH-TO/LVC-event-duration/code/io/projection_plain_verbs





