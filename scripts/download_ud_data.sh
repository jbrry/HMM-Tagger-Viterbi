#!/bin/bash

# usage: ./download_ud_data.sh UD_English-LinES en_lines

DATA_DIR="data"
DATASET=$1
DATASET_ID=$2

cd data
git clone "https://github.com/UniversalDependencies/$DATASET.git"
cd "$DATASET"

for CONLLU in $(ls | fgrep .conllu); do
    IFS=- read s1 s2 FILE_EXTENSION <<< "$CONLLU"
    IFS=. read FILE_TYPE s3 <<< "$FILE_EXTENSION"

    # take the data from columns 2 and 4 from the full file and write to a POS file
    cat "$DATASET_ID-ud-$FILE_TYPE.conllu" | cut -f 2,4 | grep -v "#" > "$DATASET_ID-ud-$FILE_TYPE.pos"
done
