#! /usr/bin/env python
""" Usage: semantic_word_projection.py --vectorspace=<vectorspace> --basis-dimension=<basis-dimension> --input-words=<input-words> --projection-words=<projection-words> --output=<output> 
    
Perform semantic projection technique of Grand et al. (2022) on individual words (for callibrating the Duration projection scale).
"""
from docopt import docopt

from pyspark.sql import SparkSession

import logging


from util.read import read_ppmi_vector_space, get_unique_tokens
from util.write import write_to_file
from util.misc import convert_to_sparse_vector, convert_to_index_value_list
from util.subspace_projection import transform_common_joint_dimensions_union
import re
import pandas as pd

import os

from scipy.sparse import csr_array, linalg, sparray
import numpy as np

ARGS = docopt(__doc__)

VECTORSPACE_PATH = ARGS['--vectorspace']
BASIS_DIMENSION = int(ARGS['--basis-dimension'])
PROJECTION_WORDS_PATH = ARGS['--projection-words']
OUTPUT_PATH = ARGS['--output']
INPUT_PATH = ARGS['--input-words']

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

    
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

    return mean_category_1 - mean_category_2, mean_category_1, mean_category_2

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
    settings_path = OUTPUT_PATH + f"/semantic_projection_settings.txt"
    write_to_file(settings_path,"".join(settings_string))

    input_words = pd.read_csv(INPUT_PATH,header=0)
    tokens = get_unique_tokens(INPUT_PATH)
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

    scale_vector, category1, category2 = calculate_scale_vector(projection_words)
    scale_vector_norm = linalg.norm(scale_vector)

    output_list = [["word","projection"]]
    not_found_list = []
    for idx,row in input_words.iterrows():
        word = row['word']
        word_vector = vectors[vectors['lemma_pos']==word]['ppmi_sparse'].values[0]

        projection = scalar_projection(word_vector,scale_vector,scale_vector_norm)

        output_list.append([word,str(projection)])

    output_string = "\n".join([",".join(line) for line in output_list])
    not_found_string = ",".join(not_found_list)

    write_to_file(OUTPUT_PATH + f"/semantic_projections.txt", output_string)

    with open(OUTPUT_PATH + f"/semantic_projection_scale.txt",'w') as f:
        f.write('category|ppmi\n')
        f.write(f"category1|{convert_to_index_value_list(category1)}\n")
        f.write(f"category2|{convert_to_index_value_list(category2)}\n")        

        

    # with decade
    # python explore_vectorspace/semantic_word_projection.py --vectorspace=PATH-TO/LVC-event-duration/code/io/ppmi.txt --basis-dimension=2000 --input-words=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration/calibration_words.txt --projection-words=PATH-TO/LVC-event-duration/code/io/duration_liu_chersoni.csv --output=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration

    # without decade
    # python explore_vectorspace/semantic_word_projection.py --vectorspace=PATH-TO/LVC-event-duration/code/io/ppmi.txt --basis-dimension=2000 --input-words=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration_without_decade/calibration_words.txt --projection-words=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration_without_decade/duration_liu_chersoni_NO_DECADE.csv --output=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration_without_decade

  





