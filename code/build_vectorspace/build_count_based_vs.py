#! /usr/bin/env python
""" Usage: build_count_based_vs.py --input=<input> --freqs=<freqs> --output=<output> [options]

Options:
--basis-dim=<basis-dim>                         Dimension of basis. [default: 2000]
--exclude-first=<exclude-first>                 Number of most frequent words to exclude from basis. [default: 50]
--word-window=<word-window>                     Word window on left and right side (n_x_n). [default: 5]
--freq-threshold=<freq-threshold>               Minimum frequency to keep a word in the vocabulary. [default: 50]
--min-sentence-length=<min-sentence-length>     Minimum sentence length to process the sentence. [default: 5]
--vocabulary-size=<vocabulary-size>             Maximum vocabulary size. [default: 50000]

    
Build co-occurrence count-based vector space.
Input: pre-processed corpus with for each sentence a line of ""lemma_pos"tokens
Output: CSV file with a line for each word in the vocabulary: lemma_pos|[(basis_index_i,count_i),(basis_index_j,count_j)]
"""

from docopt import docopt
import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

from pyspark.sql import SparkSession
import pyspark.sql.functions as psf

from pyspark.sql.types import *

import logging

import os

import scipy.sparse as sp
import numpy as np
from collections import defaultdict

from typedspark import DataSet

from pyspark.accumulators import AccumulatorParam

from util.misc import *
from util.read import read_basis, read_freqs, read_corpus, read_target_words

ARGS = docopt(__doc__)

INPUT_PATH = ARGS['--input']
FREQS_PATH = ARGS['--freqs']
OUTPUT_PATH = ARGS['--output']
os.makedirs(OUTPUT_PATH,exist_ok=True)

BASIS_DIM = int(ARGS['--basis-dim'])
EXCLUDE_FIRST = int(ARGS['--exclude-first'])
WORD_WINDOW = int(ARGS['--word-window'])
FREQ_THRESHOLD = int(ARGS['--freq-threshold'])
MIN_SENTENCE_LENGTH = int(ARGS['--min-sentence-length'])
VOCABULARY_SIZE = int(ARGS['--vocabulary-size'])

# A DefaultDict with lemma_pos,sp.lil_array entries and default value an empty splil_array.
VECTOR_SPACE = {}
# The outputstring for the final vector space output file
VECTOR_SPACE_OUTPUT = ""

#######################
## Start helper code ##
#######################

# Accumulator used for adding vectors to the vector space
class DictAccumulatorParam(AccumulatorParam):
    def zero(self,value):
        return defaultdict(lambda: sp.csr_array((1,BASIS_DIM),dtype=np.int64))
    def addInPlace(self, value1, value2):
        for k,v in value2.items():
            value1[k] +=v
        return value1

# Accumulator used for making the outputstring for the final vector space output file
class StringAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return ""
    def addInPlace(self, value1, value2):
        return value1 + value2 

"""
convert_sparse_array_to_index_count_pairs takes a sparse array as input and generates a list of [basis_index,count] pairs representing that sparse array
"""
def convert_sparse_array_to_index_count_pairs(vector):
    ignore,columns = vector.nonzero()
    res = []
    for i in columns:
        res.append([i,vector[0,i]])
    return res

"""
generate_basis_lemma_dict takes a Spark DataFrame as input and creates a Python dictionary with lemma_pos,basis_index pairs.
"""   
def generate_basis_lemma_dict(basis_df:DataSet):
    basis_pd = basis_df.toPandas()

    basis_dict = {}

    for idx,row in basis_pd.iterrows():
        basis_dict[row["lemma_pos"]] = row["basis_index"]

    return basis_dict

#####################
## Start main code ##
#####################
def process_sentence(VECTOR_SPACE,s,target_words,basis_lemma_dict):
    word_vectors = defaultdict(lambda: sp.lil_array((1,BASIS_DIM),dtype=np.int64))

    for widx,word in enumerate(s):
        if word not in target_words:
            continue

        word_vector = word_vectors[word]

        for cwidx,context_word in enumerate(s):
            if (word == context_word) or (context_word not in basis_lemma_dict.keys()) or (abs(widx-cwidx) > WORD_WINDOW):
                continue
            else:
                basis_index = basis_lemma_dict[context_word]
                word_vector[0,basis_index] +=1

    for lemma,vector in word_vectors.items():
        word_vectors[lemma] = sp.csr_array(vector)

    VECTOR_SPACE += word_vectors



'''
add_line_to_vector_space_file takes as input a lemma (LEMMA_POS) and VECTOR_SPACE_OUTPUT, a StringAccumulator, which holds all lines to be written to the vector space output file.
The output line for the lemma is first generated and then added to the VECTOR_SPACE_OUTPUT.
'''
def add_line_to_vector_space_file(lemma, VECTOR_SPACE_OUTPUT):
    # obtain the sparse vector to the corresponding lemma
    vector = VECTOR_SPACE[lemma]
    # convert the sparse array to a list pf [index,count] pairs
    res = convert_sparse_array_to_index_count_pairs(vector)
    VECTOR_SPACE_OUTPUT += "{}|{}\n".format(lemma,res)


if __name__ == '__main__':

    #https://stackoverflow.com/a/57008245
    logging.getLogger("py4j").setLevel(logging.INFO)

    settings_file_path = OUTPUT_PATH + "/build_count_based_vs_settings.txt"
    if not os.path.exists(settings_file_path):
        with open(settings_file_path, "w") as settings_file:
            for e in ARGS.items():
                settings_file.write("{} {}\n".format(e[0], e[1]))
    

    spark = SparkSession.builder \
        .config("spark.driver.memory", "10g")\
        .getOrCreate()

    freqs = read_freqs(spark, FREQS_PATH).persist()

    basis = read_basis(freqs, BASIS_DIM, EXCLUDE_FIRST).persist()
    basis.write.option('delimiter', ',') \
        .option('header', 'True') \
        .mode('ignore') \
        .csv(OUTPUT_PATH + '/CBVS_basis.csv')

    basis_lemma_dict = generate_basis_lemma_dict(basis)
    basis.unpersist()
    
    target_words = read_target_words(freqs, FREQ_THRESHOLD, VOCABULARY_SIZE).persist()
    target_words.write.option('delimiter', ',') \
        .option('header', 'true') \
        .mode('ignore') \
        .csv(OUTPUT_PATH +'/CBVS_target_words.csv')
    # make a list of all target words
    target_words_list = target_words.select('lemma_pos').agg(psf.collect_set("lemma_pos")).collect()[0][0]

    freqs.unpersist()
    target_words.unpersist()

    corpus = read_corpus(spark,INPUT_PATH).persist()

    VECTOR_SPACE = spark.sparkContext.accumulator(defaultdict(lambda: sp.csr_array((1,BASIS_DIM),dtype=np.int64)), DictAccumulatorParam())

    corpus.rdd.map(lambda r: r.value.rstrip(' ').split(' '))\
        .map(lambda l: (len(l),l))\
        .filter(lambda r: r[0]>MIN_SENTENCE_LENGTH)\
        .map(lambda r: r[1])\
        .foreach(lambda s: process_sentence(VECTOR_SPACE, s,target_words_list,basis_lemma_dict))
    
    corpus.unpersist()
    

    VECTOR_SPACE_OUTPUT = spark.sparkContext.accumulator("", StringAccumulatorParam())

    vector_space_file_path = OUTPUT_PATH + "/VS.txt"
    if not os.path.exists(vector_space_file_path):
        VECTOR_SPACE = VECTOR_SPACE.value # set variable to value, because value cannot be accessed within foreach on next line
        target_words.rdd.map(lambda r: (r[0],r[1])).sortByKey().foreach(lambda r: add_line_to_vector_space_file(r[1], VECTOR_SPACE_OUTPUT))

        # write all lines to vector space output file
        with open(vector_space_file_path, "w") as vs_file:
            vs_file.writelines(VECTOR_SPACE_OUTPUT.value)


 
    # BNC + BNC2014Spoken

    # python build_vectorspace/build_count_based_vs.py --input PATH-TO/LVC-event-duration/data/no_ART-PUNC-UNC-TRUNC-INTERJ --freqs PATH-TO/LVC-event-duration/data/ANON_NUM_-BNC_COMBINED-freq-no_ART-PUNC-UNC-TRUNC-INTERJ_pipeline.txt --output PATH-TO/LVC-event-duration/code/io


   




