#! /usr/bin/env python


from collections import defaultdict 
from scipy.sparse import lil_array
from util.subspace_projection import transform_common_joint_dimensions_union



# """
# based on Gijs Wijnholds: https://github.com/gijswijnholds/compdisteval/blob/master/code/util/util.py
# """


def BNC_to_Wacky_pos(tag):
      if tag == "ADJ":
            return "JJ"
      elif tag == "SUBST":
            return "NN"
      elif tag == "VERB":
            return "VB"
      elif tag == "ADV":
            return "RB"
      
def BNC_to_wacky_lemma_pos(lemma_pos):
    lemma, pos = lemma_pos.split('_')
    return lemma + "_" + BNC_to_Wacky_pos(pos) 


def convert_to_sparse_vector(basis_dim, basis_index_count):
        cols = basis_dim
        vector = lil_array((1,cols))
        for basis_index, count in basis_index_count:
            vector[0,int(basis_index)] = count
        return vector

'''
convert dok_matrix to [[index_i,value_i]]
'''
def convert_to_index_value_lists(ppmi_matrix):
        rs, cs = ppmi_matrix.nonzero()
        idc =   [*zip(rs,cs)]
        index_ppmi = defaultdict(lambda: [])
        for r,c in idc:
            index_ppmi[r]= index_ppmi[r] + [[float(c),float(ppmi_matrix[r,c])]]
        return index_ppmi
'''
convert sparse vector to a list of [[index_i,value_i]]
'''
def convert_to_index_value_list(ppmi_sparse_vector):
        rs, cs = ppmi_sparse_vector.nonzero()
        idc =   [*zip(rs,cs)]
        index_value_list = []
        for r,c in idc:
            index_value_list += [[float(c),float(ppmi_sparse_vector[r,c])]]
        return index_value_list


def pd_vectorspace_to_sparse(vectors,dimensions,top_n_dimensions=None):
      ppmi_vectors = {}
      for idx,row in vectors.iterrows():
            sparse = convert_to_sparse_vector(dimensions,row['ppmi'])
            ppmi_vectors[row['lemma_pos']] = sparse

      if top_n_dimensions:
        ppmi_vectors, nonzero_dimensions = transform_common_joint_dimensions_union(ppmi_vectors,top_n_dimensions)
      
      vectors['ppmi_sparse'] = vectors['lemma_pos'].map(ppmi_vectors)
      vectors = vectors.drop(columns=['ppmi'])
      return vectors