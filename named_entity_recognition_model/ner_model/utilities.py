import math
import pickle
import re
import numpy
from numpy.core.defchararray import find
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

ITERATIONS = 20
DROPOUT = 0.1
LEARN_RATE = 0.001

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

                # Incase word not found in the sentence
                if start_index == -1:
                    print("-1 error")
                    print(data['text'][index])
                    break

                # TODO: Test this extra check when more cleaner data is available.
                # There should be a blank space after every tagged word (assuming it's not the last word of
                # the sentence), otherwise it implies this tagged word is currently a part of next word,
                # and so this isn't what we were looking for. So keep looking ahead.
                # Example: "guitar" exists in "guitarist", so skip this.

                # print(data['text'][index].lower())
                # print(start_index, end_index)
                # print(data['text'][index].lower()[end_index])

                # while True:
                #     if len(data['text'][index].lower()) != end_index:
                #         if data['text'][index].lower()[end_index] != " ":
                #             start_index = find(data['text'][index].lower(), word_pair_list[0],
                #                                start=end_index + 1).astype(numpy.int64)
                #             start_index = start_index + 0
                #             end_index = start_index + len(word_pair_list[0])
                #         else:
                #             break
                #     else:
                #         break

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

    # save_list_to_txt(DATA)
    return DATA


def save_list_to_txt(data, keyword):
    with open(keyword + ".txt", 'w') as f:
        for item in data:
            f.write("%s\n" % str(item))


def save_list_to_pickle(list, name):
    # If the directory does not exist, create it
    if not os.path.exists("data"):
        os.makedirs("data")

    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(list, f)


def load_list_from_pickle(filename):
    with open("data/" + filename + '.pkl', 'rb') as f:
        list = pickle.load(f)
    return list


def split_data(DATA):
    random.shuffle(DATA)

    # Randomly pull out 10 % segments of DATA for test + eval
    test_length = math.floor((10 / 100) * len(DATA))
    TEST = DATA[:test_length]

    random.shuffle(TEST)
    # Randomly pull out 50 % segments of TEST_DATA for EVAL_DATA
    eval_length = math.floor((50 / 100) * len(TEST))
    EVAL_DATA = TEST[:eval_length]
    TEST_DATA = TEST[eval_length:len(TEST)]

    # for text, annotations in TEST_DATA:
    #     print(text)
    #     print(annotations)

    TRAIN_DATA = DATA[test_length:len(DATA)]
    print("\n")

    # for text, annotations in TRAIN_DATA:
    #   print(text)
    #   print(annotations)

    print("\nTotal sentences: ", len(DATA))
    print("Length of train data: ", len(TRAIN_DATA))
    print("Length of evaluation data: ", len(EVAL_DATA))
    print("Length of test data: ", len(TEST_DATA))

    return TRAIN_DATA, EVAL_DATA, TEST_DATA


def plot_graph(title, keyword, precision=None, recall=None, fscore=None):
    my_dpi = 200
    plt.rcParams['figure.figsize'] = 10, 5
    plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), dpi=my_dpi)
    x = list(range(1, ITERATIONS + 1))
    legend_to_show = ()
    if precision is not None:
        plt.plot(x, precision, color='red', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='red', markersize=2)
        legend_to_show += ("precision",)
    if recall is not None:
        plt.plot(x, recall, color='blue', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='blue', markersize=2)
        legend_to_show += ("recall",)
    if fscore is not None:
        plt.plot(x, fscore, color='green', linestyle='solid', linewidth=1,
                 marker='o', markerfacecolor='green', markersize=2)
        legend_to_show += ("fscore",)
    plt.gca().legend(legend_to_show, loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    plt.title(title + " PRF Scores [" + keyword + "]")

    # If the directory does not exist, create it
    if not os.path.exists("img"):
        os.makedirs("img")

    plt.savefig("img/plot_" + title + "_" + keyword + ".png", format="png", dpi=my_dpi)
    # plt.show()


def draw_prf_graph(train_scores, keyword="", overall=True, instr=True, qlty=True, edge=True):
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

    if overall is True:
        plot_graph(title=keyword, keyword="overall", precision=precision, recall=recall,
                   fscore=fscore)
    if qlty is True:
        plot_graph(title=keyword, keyword="qlty", precision=qlty_p, recall=qlty_r, fscore=qlty_f)
    if instr is True:
        plot_graph(title=keyword, keyword="instr", precision=instr_p, recall=instr_r,
                   fscore=instr_f)
    if edge is True:
        plot_graph(title=keyword, keyword="edge", precision=edge_p, recall=edge_r, fscore=edge_f)


def draw_train_eval_compare_graph(train_scores, eval_scores):
    train_fscore = []
    eval_fscore = []

    for i, train_score in enumerate(train_scores):
        for key, cat in train_score.items():
            if key == "ents_f": train_fscore.append(cat)

    for i, eval_score in enumerate(eval_scores):
        for key, cat in eval_score.items():
            if key == "ents_f": eval_fscore.append(cat)

    my_dpi = 200
    plt.rcParams['figure.figsize'] = 10, 5
    plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), dpi=my_dpi)
    x = list(range(1, ITERATIONS + 1))

    poly_order = 4

    plt.plot(x, train_fscore, color='red', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='red', markersize=2)
    train_reg_line = np.polyfit(np.array(x), np.array(train_fscore), poly_order)
    p = np.poly1d(train_reg_line)
    plt.plot(x, p(x), color='red', linestyle='--', linewidth=0.6,
             marker='o', markerfacecolor='red', markersize=1, label='_nolegend_')

    plt.plot(x, eval_fscore, color='blue', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=2)
    eval_reg_line = np.polyfit(np.array(x), np.array(eval_fscore), poly_order)
    p = np.poly1d(eval_reg_line)
    plt.plot(x, p(x), color='blue', linestyle='--', linewidth=0.6,
             marker='o', markerfacecolor='blue', markersize=1, label='_nolegend_')

    plt.gca().legend(("train", "eval"), loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title("F-Score vs Epochs")
    plt.ylim(0.00, 1.00)
    plt.savefig("img/plot_fscore_train_vs_eval.png", format="png", dpi=my_dpi)
    plt.show()


def plot_training_loss_graph(losses, title):
    my_dpi = 200
    plt.rcParams['figure.figsize'] = 10, 5
    plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), dpi=my_dpi)
    x = list(range(1, ITERATIONS + 1))
    plt.plot(x, losses, color='blue', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='green', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)

    # If the directory does not exist, create it
    if not os.path.exists("img"):
        os.makedirs("img")

    plt.savefig("img/plot_loss_training" + ".png", format="png", dpi=my_dpi)
    plt.show()
    save_list_to_txt(losses, "img/losses_list")
