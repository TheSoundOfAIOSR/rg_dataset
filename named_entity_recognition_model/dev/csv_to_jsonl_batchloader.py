"""
Load, preprocess and save (to JSONL) the CSV data in batches
Basically calls the function load_cleaned_data() in a loop, for every batch to be created
"""

import os
import pandas as pd
from pandas import DataFrame
from spacy.util import minibatch
import json

from csv_to_jsnol_loader import load_cleaned_data, \
    TRAIN_DATA_PATH, TEST_CONTENT_DATA_PATH, TEST_CONTEXT_DATA_PATH, TEST_UNSEEN


def process_in_batches(DATA_PATH, out_file_name, div_size):
    # Create assets directory if it doesn't already exist
    if not os.path.exists("assets"):
        os.makedirs("assets")
    if not os.path.exists("assets/processed_jsonl_files"):
        os.makedirs("assets/processed_jsonl_files")

    # Read the CSV file as Pandas df
    col_names = ['text', 'entities']
    data = pd.read_csv(DATA_PATH, names=col_names, usecols=[0, 1])

    # Shuffle the whole train data
    data = data.sample(frac=1).reset_index(drop=True)

    # Calculate size of each of the `div` batches
    tot_size = len(data)
    div = div_size
    num_groups = int(tot_size / div)
    print(f"Size of each part: {num_groups}\n")

    # Divide the data into batches
    data_batches = minibatch(data.to_numpy().tolist(), size=num_groups)

    # Process each batch one by one, and save its result in a seperate jsonl file
    for count, data_batch in enumerate(data_batches):
        # if count < 10:
        #     # Continue from the desired last batch
        #     continue

        # Convert the data_batches back to Pandas
        data_df = DataFrame(data_batch, columns=col_names)

        # Parameter `data_path` here is not used in the called function,
        # as it is essential only when data is NOT being loaded in batches.
        TRAIN_DATA = load_cleaned_data(data_path=DATA_PATH,
                                       data_pd=data_df)

        with open(f"assets/processed_jsonl_files/{out_file_name}{count}.jsonl", 'w') as f:
            for entry in TRAIN_DATA:
                json.dump(entry, f)
                f.write('\n')

        print(f"Batch {count} procesed and saved.")

        del TRAIN_DATA
        del data_df


if __name__ == '__main__':
    div_size = 100
    process_in_batches(TEST_CONTEXT_DATA_PATH, "TEST_CONTEXT", div_size)
