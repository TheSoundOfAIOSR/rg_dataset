import re
import numpy
from numpy.core.defchararray import find
import random
import pandas as pd
from pathlib import Path


def load_cleaned_data(data_path='processed_data.csv'):
    """
    Go through every sentence's all word-tag pair (except "NONE")
    and calculate the start and end index.
    After getting the (start, end) pair, check if this pair was already calculated
    (i.e., either the start_index, OR end_index, OR both are matching with the ones in list),
    and if so, discard the pair and continue calculating again, skipping over the one discarded.
    :return: DATA
    """
    col_names = ['text', 'entities']

    data = pd.read_csv(data_path, names=col_names)
    entity_list = data.entities.to_list()

    DATA = []

    for index, ent in enumerate(entity_list):
        if ent == "split_sentences":
            continue

        ent = ent.split("), (")
        ent[0] = re.sub("[([]", "", ent[0])
        ent[-1] = re.sub("[)]]", "", ent[-1])

        # Initialize index list, to store pairs of (start, end) indices
        indices_list = [(-1, -1), (-1, -1)]

        annot_list = []
        start_index = 0
        end_index = 0

        # Analyze current "split_sentences"'s all word-pairs
        for index_ent, word_pair in enumerate(ent):
            # Split the word and its pair
            word_pair_list = word_pair.split("'")[1::2]
            if word_pair_list[1] != "NONE":

                # Remove any leading or beginning blank space
                word_pair_list[0] = word_pair_list[0].strip()

                start_index = find(data['text'][index].lower(), word_pair_list[0]).astype(numpy.int64)
                start_index = start_index + 0
                end_index = start_index + len(word_pair_list[0])

                # Doesn't happen, just for a check
                if start_index == -1:
                    print("-1 error")
                    print(data['text'][index])
                    break

                # Check if this start_index and/or end_index is already in the list:
                # (To prevent overlapping with already tagged words)
                while True:
                    if ((start_index, end_index) in indices_list) or (
                            end_index in [i[1] for i in indices_list]) or (
                            start_index in [i[0] for i in indices_list]):
                        start_index = find(data['text'][index].lower(), word_pair_list[0],
                                           start=end_index + 1).astype(
                            numpy.int64)
                        start_index = start_index + 0
                        end_index = start_index + len(word_pair_list[0])

                    else:
                        indices_list.append((start_index, end_index))
                        break

                annot_list.append((start_index, end_index, word_pair_list[1]))

        DATA.append((data['text'][index].lower(), {"entities": annot_list}))
        # print(indices_list)

    return DATA


def split_data(DATA):
    random.shuffle(DATA)

    # First 5 elements form test data after shuffling
    TEST_DATA = DATA[:5]

    # for text, annotations in TEST_DATA:
    #     print(text)
    #     print(annotations)

    TRAIN_DATA = DATA[5:len(DATA)]
    print("\n")

    # for text, annotations in TRAIN_DATA:
    #   print(text)
    #   print(annotations)

    print("\nLength of test data: ", len(TEST_DATA))
    print("Length of train data: ", len(TRAIN_DATA))

    return TRAIN_DATA, TEST_DATA