#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate elldisco

PATH_TO=YOUR_PATH_HERE

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/RG_word.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > RG_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/MC_word.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > MC_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/wordsim353_sim_rel/wordsim353combined.txt --dataset-pos NN --word1-col word1 --word2-col word2 --humsim-col humsim > WS353_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/MEN/MEN_dataset_lemma_form_full --dataset-pos lemma-pos_lemma-pos --word1-col word1 --word2-col word2 --pos-col col1-pos_col2-pos --humsim-col humsim > MEN_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/MEN/MEN_dataset_lemma_form_full_ADJ --dataset-pos lemma-pos_lemma-pos --word1-col word1 --word2-col word2 --pos-col col1-pos_col2-pos --humsim-col humsim > MEN_ADJ_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/MEN/MEN_dataset_lemma_form_full_NOUNS --dataset-pos lemma-pos_lemma-pos --word1-col word1 --word2-col word2 --pos-col col1-pos_col2-pos --humsim-col humsim > MEN_NOUNS_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/MEN/MEN_dataset_lemma_form_full_VERB --dataset-pos lemma-pos_lemma-pos --word1-col word1 --word2-col word2 --pos-col col1-pos_col2-pos --humsim-col humsim > MEN_VERB_$(date +%F_%T).txt
    
python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/SimLex-999/SimLex-999.txt --dataset-pos pos-idx-2 --word1-col word1 --word2-col word2 --pos-col POS --humsim-col SimLex999 > SL999_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/SimLex-999/SimLex-999_ADJ.txt --dataset-pos pos-idx-2 --word1-col word1 --word2-col word2 --pos-col POS --humsim-col SimLex999 > SL999_ADJ_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/SimLex-999/SimLex-999_NOUNS.txt --dataset-pos pos-idx-2 --word1-col word1 --word2-col word2 --pos-col POS --humsim-col SimLex999 > SL999_NOUNS_$(date +%F_%T).txt

python validate_vectorspace/validate_vector_space.py --vectorspace $PATH_TO/LVC-event-duration/code/io/ppmi.txt --dataset $PATH_TO/LVC-event-duration/code/validate_vectorspace/validation_data/SimLex-999/SimLex-999_VERBS.txt --dataset-pos pos-idx-2 --word1-col word1 --word2-col word2 --pos-col POS --humsim-col SimLex999 > SL999_VERBS_$(date +%F_%T).txt

# ./explore_vectorspace/wordsimtasks.sh