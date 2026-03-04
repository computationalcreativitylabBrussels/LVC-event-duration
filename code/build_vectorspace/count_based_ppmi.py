#! /usr/bin/env python
""" Usage: count_based_ppmi.py --vectorspace <vectorspace> --output <output>
    
Compute PPMI of CBVS
"""
from docopt import docopt

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, LongType, FloatType
from typedspark import DataSet
import pyspark.sql.functions as psf

import logging


from util.read import read_vector_space
from util.misc import convert_to_index_value_lists

import os

from scipy.sparse import lil_matrix, lil_array, dok_matrix
import numpy as np

ARGS = docopt(__doc__)

OUTPUT_PATH = ARGS['<output>']

BASIS_DIM = 2000

def convert_to_sparse_vector(basis_dim, id, basis_index_count):
        cols = basis_dim + 1
        vector = lil_array((1,cols))
        vector[0,basis_dim]=id # put the ID at the back of the vector
        for basis_index, count in basis_index_count:
            vector[0,basis_index] = count
        return [vector]


def convert_to_matrix(vector_space: DataSet):
    vs_schema = StructType([StructField('lemma_pos', StringType(), False),
                         StructField('row_id', LongType(), False)])
    
    vs_rdd_with_IDs = vector_space.rdd.zipWithUniqueId()

    vs_df = vs_rdd_with_IDs.map(lambda row: (row[0].__getitem__('lemma_pos'), row[1]))\
                          .toDF(vs_schema)
    
    matrix = lil_matrix((vector_space.count(),BASIS_DIM+1))

    # https://stackoverflow.com/questions/31567989/apache-spark-how-to-create-a-matrix-from-a-dataframe
    vectors = vs_rdd_with_IDs.map(lambda row: (row[1], convert_to_sparse_vector(BASIS_DIM,row[1],row[0].__getitem__('basis_index_count')))) \
                   .sortByKey() \
                   .flatMap(lambda row: row[1])\
                   .collect()
    
    for idx, vector in enumerate(vectors):
        matrix[idx] = vector

    return vs_df, matrix

# based on: https://stackoverflow.com/a/74118510/12497648 and https://towardsmachinelearning.org/positive-point-wise-mutual-information-ppmi/
def compute_ppmi(vector_space_matrix, df: DataSet):
    # copy original matrix
    id_vsmatrix = vector_space_matrix.copy()

    #get an array of the IDs
    row_ids = id_vsmatrix[:,-1:].toarray().transpose()[0]

    # remove last column from matrix (the unique ID)
    vector_space_matrix = vector_space_matrix.tocsr()[:,:-1] # parameter gets overwritten by matrix without the indices that were in the last column

    total = vector_space_matrix.sum()
    row_totals = vector_space_matrix.sum(1)
    column_totals = vector_space_matrix.sum(0)

    row_prob = row_totals / total
    column_prob= column_totals / total

    ppmi_matrix = dok_matrix(vector_space_matrix.shape) # matrix  is created in the dimensions of the original matrix without the indices

    for row, col in zip(*vector_space_matrix.nonzero()):
        word_context_prob = vector_space_matrix[row,col] / total
        word_prob = row_prob[row]
        context_prob = column_prob[0,col]
        pmi = np.log2(word_context_prob/(word_prob * context_prob))
        ppmi = max(pmi,0)
        ppmi_matrix[row,col] = ppmi

    ppmi_matrix = convert_to_index_value_lists(ppmi_matrix) # matrix is converted to a list of lists where each inner list is of the form [[index_i,ppmi_i]] (the indices of the last column are not included here because of line number 63!)

    ppmi_df_schema = StructType([StructField('lemma_pos', StringType(), False),
                         StructField('ppmi', ArrayType(ArrayType(FloatType())))])
    
    get_index = lambda idx: np.where(row_ids==idx)[0][0]

    ppmi_df = df.rdd.map(lambda row: (row[0], ppmi_matrix[get_index(row[1])]))\
        .toDF(ppmi_df_schema)

    return ppmi_df





if __name__ == '__main__':

    logging.getLogger("py4j").setLevel(logging.INFO)

    settings_file_path = OUTPUT_PATH + "/count_based_ppmi_settings.txt"
    if not os.path.exists(settings_file_path):
        with open(settings_file_path, "w") as settings_file:
            for e in ARGS.items():
                settings_file.write("{} {}\n".format(e[0], e[1]))

    spark = SparkSession.builder\
                        .config("spark.driver.maxResultSize", "0")\
                        .getOrCreate()

    vector_space = read_vector_space(spark, ARGS['<vectorspace>'])

    vs_dataframe, vs_matrix = convert_to_matrix(vector_space)

    ppmi = compute_ppmi(vs_matrix, vs_dataframe)

    ppmi.withColumn('ppmi', psf.col('ppmi').cast('string'))\
        .write.options(header='True', delimiter='|') \
        .mode('overwrite') \
        .csv(os.path.join(os.getcwd(), OUTPUT_PATH+"/ppmi.txt"))



    # python build_vectorspace/count_based_ppmi.py --vectorspace PATH-TO/LVC-event-duration/code/io/VS.txt --output PATH-TO/LVC-event-duration/code/io


