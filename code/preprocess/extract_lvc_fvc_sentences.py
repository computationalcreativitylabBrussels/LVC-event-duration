#! /usr/bin/env python
""" Usage: extract_lvc_fvc_sentences.py --lvcs=<lvcs> --bnc=<bnc> --bnc-no-lem-pos=<bnc-no-lem-pos> --target-words=<target-words> --output=<output>
    
Extract sentences from (combined) BNC that contain LVCs and FVCs.
"""
from docopt import docopt

import os
import glob

import re
import pandas as pd
from collections import defaultdict
from itertools import filterfalse, islice, repeat
import multiprocessing as mp
import random

from functools import reduce

import stanza


args = docopt(__doc__)
LVCS_PATH = args['--lvcs']
BNC_PATH = args['--bnc']
BNC_NO_LEM_POS_PATH = args['--bnc-no-lem-pos']
TARGET_WORDS_PATH = args['--target-words']

OUTPUT_PATH = args['--output']
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def in_vocabulary(sentence,target_words):
    return all(map(lambda word: word in target_words, sentence))

def repair_anon_tokens(sentence):
    dashes = "--"
    if  dashes in sentence:
        dashes_index = sentence.index(dashes)
        if "anon" in sentence[dashes_index + 1].lower():
            # anon,pos = sentence[dashes_index + 1].split("_")
            anon = sentence[dashes_index + 1]
            return sentence[:dashes_index] + ["--" + anon.lower()] + sentence[dashes_index + 2:]
    return sentence

def add_dictionaries(dict1,dict2):
    for k,v in dict2.items():
        dict1[k] += v
    return dict1

def reduce_list_dictionaries(dicts):
    return reduce(add_dictionaries, dicts[1:], dicts[0])

def anonymise_numerals(sentence,numerals):
    result = []
    anonymised_numerals = set()
    for word in sentence.rstrip().split(" "):
        split_numeral = word.split("_")
        if len(split_numeral) != 2:
            result.append(word)
            continue
        lemma, pos = split_numeral
        if re.search("\d+", lemma) or lemma in numerals:
            anonymised_numeral = "--anonnumeral_" + pos
            # print(lemma, anonymised_numeral)
            result.append(anonymised_numeral)
            anonymised_numerals = anonymised_numerals.union([anonymised_numeral])
        else:
            result.append(word)
    return result,anonymised_numerals


def to_list_flatten(list_of_sets):
    result = []
    for s in list_of_sets:
        for e in s:
            result.append(e)
    return result


# https://github.com/xinxinlaoshi/QuickKisses/blob/main/target_event_extraction.py
en_nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)

def process_document_chunk(start_index,sentence,lvc_examples, full_verbs):
    possible_lv = lambda w: (w.id,w.lemma) if w.lemma in lvc_examples.keys() else False
    is_passive = lambda w: w.feats and ("Voice=Pass" in w.feats)
    candidate_sentences = []
    numerals = []

    annotated_sentence = en_nlp_pipeline(sentence.strip('\n'))
    s = annotated_sentence.sentences[0]

    candidate_lv_ids = [possible_lv(w) for w in s.words]
    candidate_lv_ids = dict(filterfalse(lambda x: not x, candidate_lv_ids))

    for w in s.words:
        if w.upos == "NUM":
            numerals.append(w.lemma)
        if (w.upos == "NOUN") and (w.deprel == "obj") and (w.head in candidate_lv_ids.keys()):
            if w.lemma in lvc_examples[candidate_lv_ids[w.head]]:
                with_repaired_anon_tokens = repair_anon_tokens(list(map(lambda word: word.lemma, s.words)))
                candidate_sentences.append((start_index, (candidate_lv_ids[w.head] + "_VERB", w.lemma + "_SUBST"), with_repaired_anon_tokens))
        if w.upos == "VERB" and w.lemma in full_verbs:
            with_repaired_anon_tokens = repair_anon_tokens(list(map(lambda word: word.lemma, s.words)))
            candidate_sentences.append((start_index,w.lemma + "_VERB", with_repaired_anon_tokens))
               
    return candidate_sentences, numerals

def process_file(file_path,lvc_examples, lvc_fvc_examples, target_words):
    print(file_path)

    full_verbs = set(map(lambda verb: verb.split('_')[0], to_list_flatten(list(lvc_fvc_examples.values()))))
    candidate_sentences = defaultdict(list)
    pos_tagged_sentences = defaultdict(list)
    all_numerals = set()

    with open(file_path, "r") as f:
        for index,line in enumerate(f):
            chunk_result,numerals =  process_document_chunk(index,line,lvc_examples, full_verbs)

            for r in chunk_result:
                candidate_sentences[r[0]].append(r[1:])
            all_numerals = all_numerals.union(numerals)

    all_numeral_tokens = set()
    with open(BNC_PATH +'/'+ os.path.split(file_path)[1]) as f:
        for i, line in enumerate(f):
            if not candidate_sentences.keys():
                break
            if i in candidate_sentences.keys():
                for verb,sentence in candidate_sentences[i]:
                    with_anonymised_numerals,numeral_tokens = anonymise_numerals(line,all_numerals)

                    all_numeral_tokens = all_numeral_tokens.union(numeral_tokens)
                    vocabulary = list(all_numeral_tokens) + list(target_words)

                    if in_vocabulary(with_anonymised_numerals,vocabulary):
                        pos_tagged_sentences[verb].append(" ".join(with_anonymised_numerals))
                del candidate_sentences[i]
            
    return pos_tagged_sentences



if __name__ == "__main__":
    lvcs_data = pd.read_csv(LVCS_PATH)
    target_words = pd.read_csv(TARGET_WORDS_PATH)["lemma_pos"].to_list()

    only_lemmas = lambda l: [x.split('_')[0].strip() for x in l]
    take_lemma = lambda lemma_pos: lemma_pos.split('_')[0].strip()

    lv_col = lvcs_data["LV_POS"].apply(take_lemma)
    lvcs_data["LV_POS"] = lv_col
    nominal_col = lvcs_data["nominal_POS"].apply(take_lemma)
    lvcs_data["nominal_POS"] = nominal_col

    lvc_examples = lvcs_data.groupby(by=["LV_POS"])["nominal_POS"].apply(set).to_dict()
    lvc_fvc_examples = lvcs_data.groupby(by=["LV_POS","nominal_POS"])["FVC_POS"].apply(set).to_dict()


    res = defaultdict(list)
    input_files = [f for f in glob.glob(os.path.join(BNC_NO_LEM_POS_PATH, '*.txt'))]

    filename = lambda path: os.path.split(path)[1]



    exclude_paths = []
    # "PATH-TO/general-cbvs/output/EXCLUDE_PATHS.txt"
    with open("EXCLUDE_PATHS.txt", "r") as f:
        exclude_paths = f.read().splitlines()

    print(f"if: {len(list(set(input_files)))}, ep: {len(list(set(exclude_paths)))}")
    # input_files = list(set(map(filename, input_files)).difference(set(map(filename, exclude_paths))))
    input_files = list(set(input_files).difference(set(exclude_paths)))
    print(f"ipd: {len(input_files)}")

    k = len(input_files)
    # input_files = random.choices(input_files,k=k)
    # print(input_files)

    with open(f"input_files_choices_{k}.txt", "a") as f:
        f.write("\n")
        f.writelines("\n".join(input_files))
   

    '''lvc_examples = lvcs_data.groupby(by=["LV_POS"])["nominal_POS"].apply(set).to_dict()
    lvc_fvc_examples = lvcs_data.groupby(by=["LV_POS","nominal_POS"])["FVC_POS"].apply(set).to_dict()
    # # # print(input_files[2])
    # # # print(input_files[0])
    for f in input_files:
        r = process_file(f,lvc_examples, lvc_fvc_examples, target_words)

    print(r) 

    lv_sentences = dict()
    fv_sentences = defaultdict(str)

    for verb,sentences in r.items():
        if type(verb) == tuple:
            verb_name = "_".join(verb)
            path = OUTPUT_PATH + "/" + verb_name + ".txt"
            lv_sentences[verb] = path
        else:
            verb_name = verb
            path = OUTPUT_PATH + "/" + verb_name + ".txt"
            fv_sentences[verb_name] = path
        
        with open(path, "w") as f:
            f.writelines("\n".join(sentences))

    corresponding_paths_output = [["LV_POS", "nominal_POS", "FVC_POS","LVC_sentences_path","FVC_sentences_path"]]
    for lv, lv_path in lv_sentences.items():
        lv_pos, nominal_pos = lv
        corresponding_fv = list(lvc_fvc_examples[(take_lemma(lv_pos), take_lemma(nominal_pos))])
        corresponding_fv = corresponding_fv[0] if corresponding_fv else False
        corresponding_paths_output.append([lv_pos,nominal_pos,corresponding_fv,lv_path,fv_sentences[corresponding_fv]])

    with open(OUTPUT_PATH + "/CORRESPONDING_PATHS.csv", "w") as f:
        lines = [",".join(line) for line in corresponding_paths_output]
        f.writelines("\n".join(lines))

    exit()
'''
    # for sentence_index, sentence in r[1]:
    #     retrieve_bnc_sentence(BNC_PATH,r[0],sentence_index,sentence)

    with mp.Manager() as manager:
        with manager.Pool(processes=8) as pool:
            lvc_examples = manager.dict(lvc_examples)
            lvc_fvc_examples_dict = manager.dict(lvc_fvc_examples)
            # target_words = list(map(lambda lemma_pos: lemma_pos.split("_")[0], target_words))
            target_words = manager.list(target_words)


            # print(lvc_examples)
            results = pool.starmap_async(process_file, zip(input_files,repeat(lvc_examples), repeat(lvc_fvc_examples_dict), repeat(target_words)))

            results.wait()
            pool.close()
            pool.join()
            results = results.get()
            # print(results)
            res = reduce_list_dictionaries(results)
            
    lv_sentences = dict()
    fv_sentences = defaultdict(str)

    for verb,sentences in res.items():
        if type(verb) == tuple:
            verb_name = "_".join(verb)
            path = OUTPUT_PATH + "/" + verb_name + ".txt"
            lv_sentences[verb] = path
        else:
            verb_name = verb
            path = OUTPUT_PATH + "/" + verb_name + ".txt"
            fv_sentences[verb_name] = path
        
        with open(path, "a") as f:
            f.write("\n")
            f.writelines("\n".join(sentences))

    corresponding_paths_output = [["LV_POS", "nominal_POS", "FVC_POS","LVC_sentences_path","FVC_sentences_path"]]
    for lv, lv_path in lv_sentences.items():
        lv_pos, nominal_pos = lv
        corresponding_fv = list(lvc_fvc_examples[(take_lemma(lv_pos), take_lemma(nominal_pos))])
        corresponding_fv = corresponding_fv[0] if corresponding_fv else False
        corresponding_paths_output.append([lv_pos,nominal_pos,corresponding_fv,lv_path,fv_sentences[corresponding_fv]])

    with open(OUTPUT_PATH + "/CORRESPONDING_PATHS.csv", "a") as f:
        lines = [",".join(line) for line in corresponding_paths_output]
        f.write("\n")
        f.writelines("\n".join(lines))



    # python preprocess/extract_lvc_fvc_sentences.py --lvcs=PATH-TO/LVC-event-duration/data/LVCs/EN_LVCs_single_word_fvc.csv --bnc=PATH-TO/LVC-event-duration/data/BNC_combined/no_ART-PUNC-UNC-TRUNC-INTERJ --bnc-no-lem-pos=PATH-TO/LVC-event-duration/data/BNC_combined/NOT_LEM-no_POS-PUNC-UNC-TRUNC-INTERJ --target-words=PATH-TO/io/CBVS_target_words.csv/part-00000-dfd4c6f7-ffb0-4b1d-afab-fe3300c255dd-c000.csv --output=PATH-TO/io/lvc_fvc_sentences




