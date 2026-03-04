#! /usr/bin/env python

# based on read_basis of Gijs Wijnholds: https://github.com/gijswijnholds/compdisteval/blob/master/code/util/read.py


import shelve

from pyspark.sql.types import *
import pyspark.sql.functions as psf
from typedspark import DataSet
import pandas as pd
import pyspark.pandas as ps
import os
import re

freq_schema_4_cols = StructType([StructField("row_idx", IntegerType(), False),
                          StructField("lemma", StringType(), False),
                          StructField("pos", StringType(), False),
                          StructField('freq', IntegerType(), False)])

freq_schema_3_cols = StructType([StructField("row_idx", IntegerType(), False),
                          StructField('freq', IntegerType(), False),
                          StructField("lemma_pos", StringType(), False)])

def read_freqs(spark, freqs_path):
    with open(freqs_path, 'r') as f:
        header = f.readline().strip('\n')
        delimiter_char = '|' if '|' in header else ' '
        n_cols = len(header.split(delimiter_char))
    if n_cols == 3:
        schema = freq_schema_3_cols
    elif n_cols == 4:
        schema = freq_schema_4_cols
    freqs = spark.read.format("csv") \
        .option("header", "false") \
        .option('sep', delimiter_char) \
        .option("ignoreLeadingWhiteSpace",True) \
        .option("ignoreTrailingWhiteSpace",True) \
        .schema(schema).load(freqs_path) 
    if n_cols == 4:
        freqs = freqs.select(psf.col('row_idx'),psf.col('freq'),psf.concat_ws('_',psf.col('lemma'),psf.col('pos')).alias('lemma_pos'))
    return freqs


basis_schema = StructType([StructField('basis_index',IntegerType(),False),
                           StructField('lemma_pos',StringType(),False)])

def read_basis(df, dimension, ignore_first):
    return df.select(df.lemma_pos)\
        .filter(psf.rlike("lemma_pos", psf.lit(r"^(--anon|[a-zA-Z])+_[A-Z]+$")))\
        .filter(~psf.rlike("lemma_pos", psf.lit(r"_(STOP|PREP|PRON|CONJ)")))\
        .filter(~psf.rlike("lemma_pos", psf.lit(r"^[a-z]_[A-Z]+$")))\
        .rdd.zipWithIndex() \
        .filter(lambda row: (ignore_first - 1 < row[1]) & (row[1] < ignore_first + dimension))\
        .zipWithIndex() \
        .map(lambda row: (row[1], row[0][0].lemma_pos)) \
        .toDF(basis_schema)
 
target_words_schema = StructType([StructField('voc_index',IntegerType(),False),
                           StructField('lemma_pos',StringType(),False)])

def read_target_words(df, freq_threshold, vocabulary_size):
    return df.filter(df.freq >= freq_threshold) \
            .select('lemma_pos').rdd.zipWithIndex() \
            .map(lambda row: (row[1], row[0].lemma_pos))\
            .toDF(target_words_schema)\
            .filter(psf.col("voc_index") < vocabulary_size)

def read_corpus(spark, path) -> DataSet:
    return spark.read.text(path, lineSep="\n")

vector_space_schema = StructType([
    StructField("lemma_pos", StringType(), False),
    StructField('basis_index_count', StringType(), False)])

def read_vector_space(spark, path) -> DataSet:
    return spark.read.options(header=False, delimiter='|')\
            .schema(vector_space_schema)\
            .csv(path)\
            .withColumn('basis_index_count', psf.from_json('basis_index_count', ArrayType(ArrayType(LongType())))) 

vector_space_ppmi_schema_2_cols = StructType([StructField("lemma_pos", StringType(), False),
                        StructField('ppmi', StringType(), False)])

vector_space_ppmi_schema_3_cols = StructType([StructField("lemma", StringType(), False),
                        StructField("pos", StringType(), False),
                        StructField('ppmi', StringType(), False)])

def read_ppmi_vector_space(spark, path) -> DataSet:
    first_csv_file = list(filter(lambda f: f.split(".")[-1]=="csv", os.listdir(path)))[0]
    with open(f"{path}/{first_csv_file}", 'r') as f:
        header = f.readline().strip('\n')
        delimiter_char = '|' if '|' in header else '\t'
        n_cols = len(header.split(delimiter_char))
    if n_cols == 2:
        delimiter_char = '|'
        schema = vector_space_ppmi_schema_2_cols
    elif n_cols == 3:
        delimiter_char = '\t'
        schema = vector_space_ppmi_schema_3_cols
    ppmi = spark.read.options(header=True, delimiter=delimiter_char)\
        .schema(schema)\
        .csv(path)\
        .withColumn('ppmi', psf.from_json('ppmi', ArrayType(ArrayType(FloatType())))) 
    if n_cols == 3:
        ppmi = ppmi.select(psf.concat_ws('_',psf.col('lemma'),psf.col('pos')).alias('lemma_pos'),psf.col('ppmi'))
    return ppmi

semantic_projection_scale_schema = StructType([StructField("category", StringType(), False),
                        StructField('ppmi', StringType(), False)])

def read_semantic_projection_scale(spark,path,delimiter="|"):
    return spark.read.options(header=True, delimiter=delimiter)\
        .schema(semantic_projection_scale_schema)\
        .csv(path)\
        .withColumn('ppmi', psf.from_json('ppmi', ArrayType(ArrayType(FloatType())))) 


def get_unique_tokens(path):
    with open(path,'r') as file:
        lines = [l.strip('\n') for l in file.readlines()][1:]
        tokens = re.split(r'\s|,',' '.join(lines))
        return set(tokens)


