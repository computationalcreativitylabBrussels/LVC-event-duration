#! /usr/bin/env python
import numpy as np
from numpy.linalg import norm

# from util.misc import convert_to_vector

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

def cosine_similarity(v1,v2,eps=1e-08):
    v1 = np.array(v1,dtype=np.float64)
    v2 = np.array(v2,dtype=np.float64)
    return np.dot(v1,v2)/max((norm(v1)*norm(v2)),eps)
