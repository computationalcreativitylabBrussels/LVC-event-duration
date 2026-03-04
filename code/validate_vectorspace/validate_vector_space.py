#! /usr/bin/env python
""" Usage: validate_vector_space.py --vectorspace <ppmi-vectorspace> --dataset <dataset> --dataset-pos <dataset-pos> [--word1-col <word1-col> --word2-col <word2-col> --pos-col <pos-col> --humsim-col <humsim-col>]
    
Compute Spearman's rho correlation between model similarity and human similarity judgements on word similarity tasks.
"""
from docopt import docopt

from pyspark.sql import SparkSession
from typedspark import DataSet

import pandas as pd
import logging
import numpy as np
from numpy.linalg import norm
from scipy import stats
import re

from util.read import read_ppmi_vector_space
from util.misc import convert_to_sparse_vector
from util.subspace_projection import transform_joint_dimensions
from collections import defaultdict

BASIS_DIM = 2000


ARGS = docopt(__doc__)
WORD1_COL = ARGS["<word1-col>"]
WORD2_COL = ARGS["<word2-col>"]
HUMSIM_COL = ARGS["<humsim-col>"]
DATASET = ARGS['<dataset>']
DATASET_POS = ARGS["<dataset-pos>"]
POS_COL = ARGS["<pos-col>"]


def convert_to_vector(basis_dim, basis_index_count):
        vector = [0] * basis_dim
        for basis_index, count in basis_index_count:
            vector[int(basis_index)] = count
        return vector

def calculate_cosine_similarity(v1,v2,eps=1e-08):
    v1 = np.array(v1,dtype=np.float64)
    v2 = np.array(v2,dtype=np.float64)
    return np.dot(v1,v2)/max((norm(v1)*norm(v2)),eps)

def calculate_cossim_humsim_corr(vector_space: DataSet, target_words_df):
    cossim = []
    eucldist = []
    dotprod = []
    humsim = []
    not_found = []
    found_pairs = []
    for idx, row in target_words_df.iterrows():
        word1 = row[WORD1_COL]
        word2 = row[WORD2_COL]

        v1 = vector_space[word1]
        v2 = vector_space[word2]

        if v1 is None:
                not_found.append(word1)
        if v2 is None:
                not_found.append(word2)
        if v1 is not None and v2 is not None:
            v1 = np.array(v1.todense()[0],dtype=np.float64)
            v2 = np.array(v2.todense()[0],dtype=np.float64)
            found_pairs.append(f"{word1} {word2}")

            cosine_similarity = calculate_cosine_similarity(v1,v2)
            euclidean_distance = norm(v1-v2)
            dot_product = np.dot(v1,v2)
            
            cossim.append(cosine_similarity)
            eucldist.append(euclidean_distance)
            dotprod.append(dot_product)
            humsim.append(row[HUMSIM_COL])

    cossim_spearmans_rho = stats.spearmanr(np.array(cossim),np.array(humsim))
    eucldist_spearmans_rho = stats.spearmanr(np.array(eucldist),np.array(humsim))
    dotprod_spearmans_rho = stats.spearmanr(np.array(dotprod),np.array(humsim))

    not_found = list(set(not_found))
    words_not_found = len(not_found)
    pairs_processed = len(found_pairs)
    percent_processed = round(pairs_processed/target_words_df.shape[0]*100,2)

    print("Cossim Spearman's Rho,Cossim p-value,Eucldist Spearman's Rho,Eucldist p-value,Dotprod Spearman's Rho,Dotprod p-value,#words not found,# pairs processed,% pairs processed\n")
    print("{};{};{};{};{};{};{};{};{}\n".format(cossim_spearmans_rho.statistic,cossim_spearmans_rho.pvalue,eucldist_spearmans_rho.statistic,eucldist_spearmans_rho.pvalue,dotprod_spearmans_rho.statistic,dotprod_spearmans_rho.pvalue,words_not_found,pairs_processed,percent_processed).replace(".",","))

    print("\nCosine Similarity Spearman's Rho: {}".format(cossim_spearmans_rho))
    print("\nEuclidean Distance Spearman's Rho: {}".format(eucldist_spearmans_rho))
    print("\nDot product Spearman's Rho: {}".format(dotprod_spearmans_rho))
    print("\nNo word embedding found for {} words".format(words_not_found))
    print("\nProcessed word pairs {} pairs, {}%".format(pairs_processed, percent_processed))

    print("\nNo word embedding found for:\n\n{}".format("\n".join(not_found)))
    print("\nProcessed word pairs:\n\n{}".format("\n".join(found_pairs)))


unique_column_items = lambda df,column_name: set(df.get(column_name).to_list())

if __name__ == '__main__':
    # https://stackoverflow.com/a/57008245
    logging.getLogger("py4j").setLevel(logging.INFO)

    spark = SparkSession.builder \
        .getOrCreate()
    
    ppmi_vector_space = read_ppmi_vector_space(spark,ARGS['<ppmi-vectorspace>'])
         

    data_pos_info = DATASET_POS
    if data_pos_info in ["NN","VB"]:
        # e.g., RG, MC, WS353
        pos = "SUBST" if data_pos_info == "NN" else "VERB"
        data = pd.read_csv(DATASET, sep=r'\t+', names=[WORD1_COL,WORD2_COL,HUMSIM_COL])
        # adapt dataframe to lemma_POS format identical to vector space entries
        data[[WORD1_COL,WORD2_COL]] = data[[WORD1_COL,WORD2_COL]].map(lambda w: f"{w.lower()}_{pos}")
    elif data_pos_info == "pos-idx-2":
        # e.g., SL999
        def to_lemma_pos(lemma,pos):
            convert_to_pos_tag = lambda tag: "SUBST" if tag=="N" else ("ADJ" if tag=="A" else "VERB")
            pos = convert_to_pos_tag(pos)
            return f"{lemma.lower()}_{pos}"
        def convert_row_to_lemma_pos(word1,word2,pos):
             return [to_lemma_pos(word1,pos),to_lemma_pos(word2,pos)]
       
        data = pd.read_csv(DATASET, sep=r'\t+', header=0)[[WORD1_COL,WORD2_COL,POS_COL,HUMSIM_COL]]
        # adapt dataframe to lemma_POS format identical to vector space entries
        data = data.apply(lambda row: convert_row_to_lemma_pos(row[WORD1_COL], row[WORD2_COL], row[POS_COL]) + [row[HUMSIM_COL]], axis=1, result_type='expand')
        data.columns = [WORD1_COL,WORD2_COL,HUMSIM_COL]
    elif data_pos_info == "lemma-pos_lemma-pos":
        # e.g., MEN task
        def to_lemma_pos(old_lemma_pos):
             convert_to_pos_tag = lambda tag: "SUBST" if tag=="n" else ("ADJ" if tag=="j" else "VERB")
             lemma, pos = old_lemma_pos.split('-')
             pos = convert_to_pos_tag(pos)
             return f"{lemma.lower()}_{pos}"

        data = pd.read_csv(DATASET, sep=r'\s', names=[WORD1_COL,WORD2_COL,HUMSIM_COL])
        # adapt dataframe to lemma_POS format identical to vector space entries
        data[[WORD1_COL,WORD2_COL]] = data[[WORD1_COL,WORD2_COL]].map(lambda w: to_lemma_pos(w))

    # obtain all unique lemma_pos tokens
    tokens = unique_column_items(data,WORD1_COL).union(unique_column_items(data,WORD2_COL))

    ppmi_vector_space = ppmi_vector_space.where(ppmi_vector_space.lemma_pos.isin(tokens)).toPandas()

    ppmi_vector_space["ppmi"] = ppmi_vector_space["ppmi"].map(lambda index_counts : convert_to_sparse_vector(BASIS_DIM,index_counts))

    # convert to dictionary(lemma_POS,sparse ppmi vector)
    ppmi_vector_space = ppmi_vector_space.set_index("lemma_pos").to_dict()['ppmi']
    ppmi_vector_space = defaultdict(lambda: None, ppmi_vector_space)

    calculate_cossim_humsim_corr(ppmi_vector_space,data)

# python validate_vectorspace/validate_vector_space.py --vectorspace PATH-TO/LVC-event-duration/code/io/ppmi.txt --dataset PATH-TO/LVC-event-duration/code/validate_vectorspace/validation_data/RG_word.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > RG_$(date +%F_%T).txt

# python validate_vectorspace/validate_vector_space.py --vectorspace PATH-TO/LVC-event-duration/code/io/ppmi.txt --dataset PATH-TO/LVC-event-duration/code/validate_vectorspace/validation_data/MC_word.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > MC_$(date +%F_%T).txt

# python validate_vectorspace/validate_vector_space.py --vectorspace PATH-TO/LVC-event-duration/code/io/ppmi.txt --dataset PATH-TO/LVC-event-duration/code/validate_vectorspace/validation_data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > WS353_$(date +%F_%T).txt

# python validate_vectorspace/validate_vector_space.py --vectorspace PATH-TO/LVC-event-duration/code/io/ppmi.txt --dataset PATH-TO/LVC-event-duration/code/validate_vectorspace/validation_data/MEN/MEN_dataset_lemma_form_full --dataset-pos lemma-pos_lemma-pos --word1-col word1 --word2-col word2 --pos-col col1-pos_col2-pos --humsim-col humsim > MEN_$(date +%F_%T).txt
    
# python validate_vectorspace/validate_vector_space.py --vectorspace PATH-TO/LVC-event-duration/code/io/ppmi.txt --dataset PATH-TO/LVC-event-duration/code/validate_vectorspace/validation_data/SimLex-999/SimLex-999.txt --dataset-pos pos-idx-2 --word1-col word1 --word2-col word2 --pos-col POS --humsim-col SimLex999 > SL999_$(date +%F_%T).txt
        
