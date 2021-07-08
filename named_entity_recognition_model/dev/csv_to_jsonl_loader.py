import os
import pandas as pd
import re
import numpy
from numpy.core.defchararray import find
import json

TRAIN_DATA_PATH = "./data/augmented_dataset_2021-06-30/train.csv"
TEST_CONTENT_DATA_PATH = "./data/augmented_dataset_2021-06-30/test_content.csv"
TEST_CONTEXT_DATA_PATH = "./data/augmented_dataset_2021-06-30/test_context.csv"
TEST_UNSEEN = "./data/augmented_dataset_2021-06-30/test_unseen.csv"


def load_cleaned_data(data_path, data_pd=None):
    """
    Go through every sentence's all word-tag pair (except "NONE")
    and calculate the start and end index.
    After getting the (start, end) pair, check if this pair was already calculated
    (i.e., either the start_index, OR end_index, OR both are matching with the ones in list),
    and if so, discard the pair and continue calculating again, skipping over the one discarded.
    :return: DATA
    """
    if data_pd is None:
        col_names = ['text', 'entities']
        data = pd.read_csv(data_path, names=col_names, usecols=[0, 1])

    else:
        # Happens only when data is being loaded in batches.
        # Incoming `data_pd` is itself a pandas,
        # so just process it.
        data = data_pd

    entity_list = data["entities"].to_list()

    DATA = []

    for index, ent in enumerate(entity_list):
        if ent == "tokens":
            continue

        ent = ent.split("), (")
        ent[0] = re.sub("[([]", "", ent[0])
        ent[-1] = re.sub("[)]]", "", ent[-1])

        # Initialize index list, to store pairs of (start, end) indices
        indices_list = [(-1, -1), (-1, -1)]

        tokens_list = []
        spans_list = []

        start_index = 0
        end_index = 0

        # Analyze current "split_sentences"'s all word-pairs
        for index_ent, word_pair in enumerate(ent):
            word_pair_list = []

            # Split the word and its pair
            word_pair_list = word_pair.split("'")[1::2]

            # Remove any leading or beginning blank space
            word_pair_list[0] = word_pair_list[0].strip()

            start_index = find(data['text'][index].lower(), word_pair_list[0]).astype(numpy.int64)
            start_index = int(start_index + 0)
            end_index = int(start_index + len(word_pair_list[0]))

            # Incase word not found in the sentence
            if start_index == -1:
                print("\n-1 error")
                print("Couldn't find:")
                print(word_pair_list[0])
                print("in:")
                print(data['text'][index])
                break

            both_present = lambda: (start_index, end_index) in indices_list
            start_present = lambda: start_index in [i[0] for i in indices_list]
            end_present = lambda: end_index in [i[1] for i in indices_list]
            left_blank = lambda: data['text'][index][start_index - 1] != " "

            def right_blank():
                # return true if there is no blank space after the end_index,
                # as long as end_index is not at the end of the sentence
                if len(data['text'][index].lower()) != end_index:
                    return data['text'][index][end_index] != " "

            # Check if this start_index and/or end_index is already in the list:
            # (To prevent overlapping with already tagged words)
            flag = 0
            while True:
                if (start_index == -1 or end_index == -1):
                    flag = 1
                    break
                if (both_present()) or (start_present()) or (end_present()) or (left_blank()) or (right_blank()):

                    start_index = find(data['text'][index].lower(), word_pair_list[0],
                                       start=end_index + 1).astype(numpy.int64)
                    start_index = int(start_index + 0)
                    end_index = int(start_index + len(word_pair_list[0]))

                else:
                    indices_list.append((start_index, end_index))
                    break

            if (flag == 1):
                # Don't bother checking rest of the current sentence
                break

            # Add ALL the words and their positions to a "tokens" list
            tokens_list.append({"text": word_pair_list[0], "start": start_index, "end": end_index})

            # Add the specially tagged words to a "spans" list
            if word_pair_list[1] != "NONE":
                spans_list.append({"start": start_index, "end": end_index, "label": word_pair_list[1]})

        DATA.append(
            {"text": data['text'][index].lower(), "tokens": tokens_list, "spans": spans_list, "answer": "accept"})

    return DATA


def dump_to_jsonl(DATA, file_name):
    if not os.path.exists("assets"):
        os.makedirs("assets")

    if not os.path.exists("assets/processed_jsonl_files"):
        os.makedirs("assets/processed_jsonl_files")

    with open('assets/processed_jsonl_files/' + file_name + '.jsonl', 'w') as f:
        for entry in DATA:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    # TRAIN_DATA = load_cleaned_data(TRAIN_DATA_PATH)
    # TEST_CONTENT = load_cleaned_data(TEST_CONTENT_DATA_PATH)
    # TEST_CONTEXT = load_cleaned_data(TEST_CONTEXT_DATA_PATH)
    UNSEEN_DATA = load_cleaned_data(TEST_UNSEEN)

    # dump_to_jsonl(TRAIN_DATA, "TRAIN_DATA")
    # dump_to_jsonl(TEST_CONTENT, "TEST_CONTENT")
    # dump_to_jsonl(TEST_CONTEXT, "TEST_CONTEXT")
    dump_to_jsonl(UNSEEN_DATA, "UNSEEN_DATA")
