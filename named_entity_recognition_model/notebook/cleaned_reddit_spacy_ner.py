import spacy
from spacy import displacy
import pandas as pd
import numpy
numpy.random.seed(0)
import re
from numpy.core.defchararray import find
import random
from spacy.util import minibatch, compounding
from pathlib import Path



#Download spacy small model
# Load SpaCy model
def load_spacy():
    print("SpaCy version: ", spacy.__version__)
    nlp = spacy.load("en_core_web_sm")
    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    return ner, nlp


def load_cleaned_data():
    """
    Go through every sentence's all word-tag pair (except "NONE")
    and calculate the start and end index.
    After getting the (start, end) pair, check if this pair was already calcualted
    (i.e., either the start_index, OR end_index, OR both are matching with the ones in list),
    and if so, discard the pair and continue calculuting again, skipping over the one discarded.
    :return: DATA
    """
    col_names = ['text', 'entities']

    data = pd.read_csv(DATA_PATH, names=col_names)
    entity_list = data.entities.to_list()

    DATA = []

    for index, ent in enumerate(entity_list):
        if (ent == "split_sentences"):
            continue

        ent = ent.split("), (")
        ent[0] = re.sub("[([]", "", ent[0])
        ent[-1] = re.sub("[)]]", "", ent[-1])

        # Initilize index list, to store pairs of (start, end) indices
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
                    if ((start_index, end_index) in indices_list) or (end_index in [i[1] for i in indices_list]) or (
                            start_index in [i[0] for i in indices_list]):
                        start_index = find(data['text'][index].lower(), word_pair_list[0], start=end_index + 1).astype(
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


def train_and_test(ner, nlp, TRAIN_DATA, TEST_DATA):
    ITERATIONS = 64
    DROPOUT = 0.1

    # Adding labels to the NER
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable pipeline components that are not changed
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Train the NER model
    with nlp.disable_pipes(*unaffected_pipes):
        for iteration in range(ITERATIONS):
            # print("Iteration: ", iteration)
            # shufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=DROPOUT,  # dropout - make it harder to memorise data
                    losses=losses
                )
                # print("Losses", losses)

    # Test the model
    for example in TEST_DATA:
        print(example[0])
        doc = nlp(example[0])
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":

    DATA_PATH = './../../reddit_data_preprocessing/processed_data.csv'

    ner, nlp = load_spacy()
    DATA = load_cleaned_data()
    TRAIN_DATA, TEST_DATA = split_data(DATA)
    train_and_test(ner, nlp, TRAIN_DATA, TEST_DATA)
