import math
import re
import numpy
from numpy.core.defchararray import find
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

ITERATIONS = 40
DROPOUT = 0.1
DATA_PATH = "./../../reddit_data_preprocessing/pf_data.csv"

def load_cleaned_data(data_path=DATA_PATH):
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

    # Randomly pull out 10 % segments for test data
    test_length = math.floor((10 / 100) * len(DATA))
    TEST_DATA = DATA[:test_length]

    # for text, annotations in TEST_DATA:
    #     print(text)
    #     print(annotations)

    TRAIN_DATA = DATA[test_length:len(DATA)]
    print("\n")

    # for text, annotations in TRAIN_DATA:
    #   print(text)
    #   print(annotations)

    print("\nTotal sentences: ", len(DATA))
    print("\nLength of test data: ", len(TEST_DATA))
    print("Length of train data: ", len(TRAIN_DATA))

    return TRAIN_DATA, TEST_DATA


def draw_prf_graph(train_scores):
    precision = []
    recall = []
    fscore = []

    qlty_p = []
    qlty_r = []
    qlty_f = []

    instr_p = []
    instr_r = []
    instr_f = []

    edge_p = []
    edge_r = []
    edge_f = []

    # Extract P, R, F from train_score
    for i, train_score in enumerate(train_scores):
        for key, cat in train_score.items():
            if key == "ents_p": precision.append(cat)
            if key == "ents_r": recall.append(cat)
            if key == "ents_f": fscore.append(cat)
            if key == "ents_per_type":
                for attribute, value in cat.items():
                    if attribute == "QLTY":
                        for k, sc in value.items():
                            if k == "p": qlty_p.append(sc)
                            if k == "r": qlty_r.append(sc)
                            if k == "f": qlty_f.append(sc)
                    if attribute == "INSTR":
                        for k, sc in value.items():
                            if k == "p": instr_p.append(sc)
                            if k == "r": instr_r.append(sc)
                            if k == "f": instr_f.append(sc)
                    if attribute == "EDGE":
                        for k, sc in value.items():
                            if k == "p": edge_p.append(sc)
                            if k == "r": edge_r.append(sc)
                            if k == "f": edge_f.append(sc)

    def plot_graph(precision, recall, fscore, title, keyword):
        my_dpi = 200
        plt.rcParams['figure.figsize'] = 10, 5
        plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), dpi=my_dpi)
        x = list(range(1, ITERATIONS + 1))
        plt.plot(x, precision, color='red', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='red', markersize=2)
        plt.plot(x, recall, color='blue', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='blue', markersize=2)
        plt.plot(x, fscore, color='green', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='green', markersize=2)
        plt.gca().legend(('precision', 'recall', 'fscore'), loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title(title)

        # If the directory does not exist, create it
        if not os.path.exists("img"):
            os.makedirs("img")

        plt.savefig("img/plot_train_prf_" + keyword + ".png", format="png", dpi=my_dpi)
        plt.show()

    plot_graph(precision, recall, fscore, title="Training Overall PRF Scores", keyword="overall")
    plot_graph(qlty_p, qlty_r, qlty_f, title="Training QLTY PRF Scores", keyword="qlty")
    plot_graph(instr_p, instr_r, instr_f, title="Training INSTR PRF Scores", keyword="instr")
    plot_graph(edge_p, edge_r, edge_f, title="Training EDGE PRF Scores", keyword="edge")
