#! /usr/bin/env python
""" Usage: preprocess_bnc.py --bnc-type <bnc-type> --bnc-path <bnc-path> --output-path <output-path> --pos-tags <pos-tags> --punctuation <punctuation> --unclassified <unclassified> --truncated <truncated> [--articles=<articles> --interjections=<interjections> --lemmatise=<lemmatise>]

Options:
--articles=<articles>               Boolean to in/exclude articles from the corpus. [default: True]
--interjections=<interjections>     Boolean to in/exclude interjections from the corpus. [default: True]
--lemmatise=<lemmatise>           Boolean indicating whether the output needs to be lemmatised. [default: True]
    
Produces files with one sentence per line.
"""
from docopt import docopt

import os, glob
import multiprocessing as mp

from lxml import etree
from time import sleep

from util.write import write_settings


args = docopt(__doc__)
CORPUS_PATH = args['<bnc-path>']
OUTPUT_PATH = args['<output-path>']

os.makedirs(OUTPUT_PATH, exist_ok=True)

get_boolean = lambda arg: True if arg in ['True', 'true'] else False

POS_TAGS = get_boolean(args['<pos-tags>'])
PUNCTUATION = get_boolean(args['<punctuation>'])
UNCLASSIFIED = get_boolean(args['<unclassified>'])
TRUNCATED = get_boolean(args['<truncated>'])
ARTICLES = get_boolean(args["--articles"])
INTERJECTIONS = get_boolean(args["--interjections"])
LEMMATISE = get_boolean(args['--lemmatise'])

BNC_TYPE = args['<bnc-type>']

get_filename = lambda path: os.path.splitext(os.path.basename(path))[0]

# BNC VS BNC2014spoken
if BNC_TYPE == 'BNC':
    SENTENCE_LABEL = 's'
    POS_LABEL = 'pos'
    LEMMA_LABEL = 'hw'
    is_punctuation = lambda e:  e.tag == 'c'
elif BNC_TYPE == 'BNC2014spoken':
    SENTENCE_LABEL = 'u'
    POS_LABEL = 'class'
    LEMMA_LABEL = 'lemma'
    is_punctuation = lambda e: e.tag == 'w' and e.get('lemma') == 'PUNC'

def process_sentence(s):
    result = []
    TRUNCATED_flag = False
    for e in s.iter():
        word_entry = ''
        if e.tag == 'trunc':
            TRUNCATED_flag = True
        elif is_punctuation(e):
            if PUNCTUATION:
                word_entry += e.text
                if POS_TAGS:
                    word_entry += '_PUNC' 
                result.append(word_entry)
            else:
                continue
        elif e.tag == 'w':
            if TRUNCATED_flag:
                TRUNCATED_flag = False
                if not TRUNCATED:
                    continue
            pos = e.get(POS_LABEL)
            if (not UNCLASSIFIED and pos == 'UNC') or (not ARTICLES and pos == 'ART') or (not INTERJECTIONS and pos == "INTERJ"):
                continue

            if LEMMATISE:
                word_entry += e.get(LEMMA_LABEL)
            else:
                word_entry += e.text.strip()

            if POS_TAGS:
                word_entry += '_' + pos
            result.append(word_entry)
    result += '\n'
    return ' '.join(result)

def process_file(file_path):
    tree = etree.parse(file_path)

    output_file_path = OUTPUT_PATH + '/' + get_filename(file_path) + '.txt'

    document = []
    for e in tree.iter(SENTENCE_LABEL):
        sentence = process_sentence(e)
        if sentence[0] != '\n':
            document.append(sentence)

    with open(output_file_path, 'w') as f:
        f.writelines(document)


def test_proc(f):
    print(len(f))


if __name__ == "__main__":
    output_parent_dir, output_filename = os.path.split(OUTPUT_PATH)
    write_settings(output_parent_dir+'/SETTINGS_'+output_filename,args)

    input_files = [f for f in glob.glob(os.path.join(CORPUS_PATH, '*.xml'))]

    with mp.Manager() as manager:
        with manager.Pool(processes=8) as pool:
            result = pool.map_async(process_file, iter(input_files))

            result.wait()

            pool.close()

            pool.join()
   

    # preprocess_bnc.py --bnc-type <bnc-type> --bnc-path <bnc-path> --output-path <output-path> --pos-tags <pos-tags> --PUNCTUATION <PUNCTUATION> --UNCLASSIFIED <UNCLASSIFIED> --TRUNCATED <TRUNCATED>

    #---------------
    # preprocess BNC:

    # python preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/plain_sentences_tagged --pos-tags True --punctuation True --unclassified True --truncated True

    # python preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/no_punctuation --pos-tags True --punctuation False --unclassified True --truncated True

    # python preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/no_punctuation_no_unclassified --pos-tags True --punctuation False --unclassified False --truncated True

    # python preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/no_punctuation_no_unclassified_no_truncated --pos-tags True --punctuation False --unclassified False --truncated False

    # python preprocess/preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/no_ART-PUNC-UNC-TRUNC-INTERJ --pos-tags True --punctuation False --unclassified False --truncated False --articles=False --interjections=False

    #--- not lemmatised no POS
    # python preprocess/preprocess_bnc.py --bnc-type BNC --bnc-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts --output-path PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/NOT_LEM-no_POS-PUNC-UNC-TRUNC-INTERJ --pos-tags False --punctuation False --unclassified False --truncated False --articles=True --interjections=False --lemmatise=False

    #---------------

    #---------------
    # preprocess bnc2014spoken

    # - plain_sentences_tagged:

    # python preprocess_bnc.py --bnc-type BNC2014spoken --bnc-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged --output-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged_plain_sentences_tagged --pos-tags True --punctuation True --unclassified True --truncated True

    # - no_punctuation: 

    # python preprocess_bnc.py --bnc-type BNC2014spoken --bnc-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged --output-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/no_punctuation --pos-tags True --punctuation False --unclassified True --truncated True

    # - no_punctuation_no_unclassified_no_truncated:

    # python preprocess_bnc.py --bnc-type BNC2014spoken --bnc-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged --output-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/no_punctuation_no_unclassified_no_truncated --pos-tags True --punctuation False --unclassified False --truncated False

    # python preprocess/preprocess_bnc.py --bnc-type BNC2014spoken --bnc-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged --output-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/no_ART-PUNC-UNC-TRUNC-INTERJ --pos-tags True --punctuation False --unclassified False --truncated False --articles=False --interjections=False

    #--- not lemmatised no POS
    # python preprocess/preprocess_bnc.py --bnc-type BNC2014spoken --bnc-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken/tagged --output-path PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/NOT_LEM-no_POS-PUNC-UNC-TRUNC-INTERJ --pos-tags False --punctuation False --unclassified False --truncated False --articles=True --interjections=False --lemmatise=False

    #---------------