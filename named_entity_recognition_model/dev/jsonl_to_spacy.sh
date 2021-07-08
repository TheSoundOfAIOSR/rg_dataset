#!/bin/bash

mkdir -p corpus

DATA_FILE="TEST_CONTEXT"

drive_path="./assets/processed_jsonl_files/$DATA_FILE"
jsonl_ext=".jsonl"

saved_path="./corpus/$DATA_FILE"
spacy_ext=".spacy"

# Get the current directory
BASEDIR=$(readlink -f $0 | xargs dirname)

for file_iter in {0..3}
do
    jsonl_drive_path="$drive_path$file_iter$jsonl_ext"
    spacy_file_path="$saved_path$file_iter$spacy_ext"

    python3 "$BASEDIR"/scripts/preprocess.py "$jsonl_drive_path" "$spacy_file_path"
done