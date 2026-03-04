#! /usr/bin/env python
""" Usage: pca2d.py --vectorspace=<vectorspace> --basis-dimension=<basis-dimension> --input-words=<input-words> --projection-scale=<projection-scale> --output=<output> 
    
Perform principal component analysis on word vectors.
"""
from docopt import docopt


# https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import rotations

from pyspark.sql import SparkSession
import numpy as np
from scipy.sparse import linalg

from util.read import read_ppmi_vector_space, read_semantic_projection_scale
from util.misc import pd_vectorspace_to_sparse, convert_to_sparse_vector

ARGS = docopt(__doc__)
VECTORSPACE_PATH = ARGS['--vectorspace']
BASIS_DIMENSION = int(ARGS['--basis-dimension'])
PROJECTION_SCALE_PATH = ARGS['--projection-scale']
OUTPUT_PATH = ARGS['--output']
INPUT_PATH = ARGS['--input-words']

def scalar_projection(target, scale, norm):
    projection = target.dot(scale.transpose())/norm
    return float(projection.todense()[0,0]) # output one singular float representing the scalar projection

if __name__ == '__main__':
    spark = SparkSession.builder\
                        .config("spark.driver.maxResultSize", "0")\
                        .getOrCreate()

    vector_space = read_ppmi_vector_space(spark, VECTORSPACE_PATH)
    input_words = pd.read_csv(INPUT_PATH,header=0)

    projection_scale = read_semantic_projection_scale(spark,PROJECTION_SCALE_PATH).toPandas()
    projection_scale_vectors = [convert_to_sparse_vector(BASIS_DIMENSION,v) for v in projection_scale["ppmi"]]

    scale = projection_scale_vectors[0] - projection_scale_vectors[1]
    scale_norm = linalg.norm(scale)
    category1_projection = scalar_projection(projection_scale_vectors[0],scale,scale_norm)
    category2_projection = scalar_projection(projection_scale_vectors[1],scale,scale_norm)

    vectors = vector_space.where(vector_space.lemma_pos.isin(input_words['word'].tolist())).toPandas()

    categories = input_words['category'].tolist()

    vectors = pd_vectorspace_to_sparse(vectors,BASIS_DIMENSION)

    word_matrix = np.array([x.todense()[0] for x in vectors['ppmi_sparse']])
    word_matrix = np.concatenate((word_matrix, *map(lambda x: x.todense(), projection_scale_vectors)), axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(word_matrix)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.axis('off')

    for category,color in [(1,"purple"),(2,"green")]:
        category_indices = [i for i,v in enumerate(categories) if v == category]
        ax.scatter(input_words['projection'][category_indices],
                   pca_result[:-2,0][category_indices],
                    c=color)


    ax.scatter([category1_projection,category2_projection],[0,0], c='r', marker="o")

    

    words = input_words['word'].tolist() 
    projection_scores = input_words['projection'].tolist()
    for i,w in enumerate(words):
        w, _ = w.split("_")
        if pca_result[i,0] > 0:
            y_offs = 1.5
        else:
            y_offs = -7
        ax.text(projection_scores[i],pca_result[i,0]+y_offs,f'{w}\n{round(projection_scores[i],2)}', ha='center', fontsize='x-large')

    for category,projection in zip(['SHORT','LONG'],[category1_projection,category2_projection]):
        ax.text(projection,1, category, fontdict={"fontweight":"bold"}, ha='center', fontsize='x-large')
  

    plt.hlines(0,min(projection_scores)-1,max(projection_scores)+1, colors="grey")
    plt.eventplot(np.array(projection_scores)[:,np.newaxis], orientation='horizontal',
                  linelengths=pca_result[:-2,0],
                  lineoffsets=[*map(lambda x: 0.5*x, pca_result[:-2,0])],
                  linestyles='dotted')
    plt.xticks([])
    plt.yticks([])

    plt.show()


    # python explore_vectorspace/pca_2d.py --vectorspace=PATH-TO/LVC-event-duration/code/io/ppmi.txt --basis-dimension=2000 --input-words=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration/semantic_projections_pca_input.txt --projection-scale=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration/semantic_projection_scale.txt --output=PATH-TO/LVC-event-duration/code/io/projection_calibration/calibration/

