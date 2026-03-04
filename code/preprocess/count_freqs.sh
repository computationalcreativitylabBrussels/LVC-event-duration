#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Generate a file with the frequencies of each lemma in the given corpus."
   echo
   echo "Syntax: count_freqs.sh input output"
   echo "input     Path to input file."
   echo "input format: preprocessed corpus existing of word_pos tokens separated by spaces"
   echo "output    Path to output file."
   echo "output format: line_number,count,lemma_pos"
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
last=${@:$#} # last parameter

echo $last

other=${*%${!#}} # all parameters except the last



#cat $other | ggrep -oP  '\K(\S+)(?=\s)' | sort | uniq -c | sort -k1 -n -r | awk '{a[$2] += $1} END{for (i in a) print a[i], i}' | sort -k1 -n -r | nl > $last

#gsed -e 's/[[:space:]]\{1,\}/,/g' $2 | gsed -e 's/^,//g' > $2+'_'

cat $other | ggrep -oP  '\K(\S+)(?=\s)' | sort | uniq -c | sort -k1 -n -r | awk '{a[$2] += $1} END{for (i in a) print a[i], i}' | sort -k1 -n -r | nl | gsed -e 's/[[:space:]]\{1,\}/|/g' | gsed -e 's/^|//g' > $last

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/no_punctuation/* PATH-TO/LVC-event-duration/data/BNC/SpokenTexts_preproc_cbvs/total_counts_no_punctuation.txt

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/no_punctuation/* PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/total_counts_no_punctuation.txt

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/no_punctuation/* PATH-TO/LVC-event-duration/data/bnc2014spoken-xml/spoken_preproc/no_punctuation/* PATH-TO/LVC-event-duration/data/BNC_COMBINED_no_punctuation_counts.txt

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/BNC_combined/* PATH-TO/LVC-event-duration/data/BNC_COMBINED_no_punctuation_counts_2.txt

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/BNC_combined/no_ART-PUNC-UNC-TRUNC-INTERJ/* PATH-TO/LVC-event-duration/data/BNC_combined/BNC_COMBINED-freq-no_ART-PUNC-UNC-TRUNC-INTERJ.txt

# ./count_freqs.sh PATH-TO/LVC-event-duration/data/BNC_combined/no_ART-PUNC-UNC-TRUNC-INTERJ/* PATH-TO/LVC-event-duration/data/BNC_combined/BNC_COMBINED-freq-no_ART-PUNC-UNC-TRUNC-INTERJ_pipeline.txt

