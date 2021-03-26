import pandas as pd
import re
import numpy as np
from collections import Counter

DATA_PATH = './doccano_data/project_1_dataset.jsonl'
OUTPUT_PATH = './reddit_data_preprocessing/processed_data.csv'


def fetch_data(data_path=DATA_PATH):

    '''fetch the data project_1_dataset.jsonl from the doccano_data repository

    :param data_path: path to the data JSONL file
    :return: dataframe containing all the data with columns (id, text, annotations, meta, annotation_approver, comments)
    '''

    data = pd.read_json(path_or_buf=data_path, lines=True)
    return data

def transform_labels(data):
    for idx, row in data.iterrows():
        for dic in row['annotations']:
            if dic['label'] == 1:
                dic['label'] = 'INSTR'
            elif dic['label'] == 2:
                dic['label'] = 'QLTY'
    return data

def split_data(data):
    ''' split the data into a labeled set (seed) and an unlabeled set

    :param data: data containing labeled and unlabeled data
    :return: dataframe called seed containing all the labeled data and unlabelled_dataset containing unlabelled data
    '''
    column_names = data.columns
    important_columns = ['text', 'annotations']

    seed = pd.DataFrame(columns=important_columns)
    unlabelled_dataset = pd.DataFrame(columns=important_columns)

    for idx, row in data.iterrows():
        if not row['annotations'] :
            unlabelled_dataset = unlabelled_dataset.append(row[important_columns], ignore_index=True)
        else :
            seed = seed.append(row[important_columns], ignore_index=True)

    return seed, unlabelled_dataset

def remove_special_chars(x):
    ''' remove URLs and special chars

    :param x: text column of the seed dataset
    :return: clean text column
    '''
    x = re.sub(r'\\n', ' ', x)
    x = re.sub(r'\\u', ' ', x)
    x = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', x)
    x = re.sub('[^-A-Za-z0-9]', ' ', x)
    return ' '.join(x.split())


def get_unique_words(splitted_sentences):
    ''' get pairs of words and their occurences and a list of unique words

    :param data_column: the column containing the text to split
    :return: tuple (word, count) and list of unique words
    '''

    ctr = Counter()

    words = [item[0] for sublist in splitted_sentences for item in sublist]

    for word in words:
        ctr[word] += 1

    unique_words = set(words)

    return ctr, unique_words


def process_data(data):
    ''' split text rows into tuples (word, label) where label can be either :
       {0 : non annotated word, 1 : instrument, 2 : quality/timbre}
        E.g. [('Example', 0), ('I', 0), ('would', 0), ('like', 0), ('to', 0),
              ('find', 0), ('a', 0), ('dark', 2), ('sounding', 0), ('piano', 1)]

    :param data: the seed dataset
    :return: processed data
    '''

    data['tagged_words'] = ''
    data['split_sentences'] = ''

    # extract annotated words as tuples (word, label) from the text column using the annotation column
    # and add them to the tagged words column
    for idx, row in data.iterrows():
        tuple_list = []
        for dic in row['annotations']:
            tuple_list.append((row['text'][dic['start_offset'] : dic['end_offset']], dic['label']))
        row['tagged_words'] = dict(tuple_list)


    # clean text data
    data['text'] = data['text'].apply(remove_special_chars)

    # split text into a list of tuples (word, label) where the word is the word in each text
    # and the label is either 3 (if it's not annotated) or 1-2 if it's annotated
    # for idx, row in data.iterrows():
    #     word_tuples = []
    #     for word in row['text'].split():
    #         if word not in row['tagged_words'].keys() :
    #             word_tuples.append((word.lower(),'NONE'))
    #         else:
    #             word_tuples.append((word.lower(), row['tagged_words'][word]))
    #     row['split_sentences'] = word_tuples

    for idx, row in data.iterrows():

        positions = []
        for key in row['tagged_words'].keys():
            position = row['text'].find(key)
            positions.extend((position, position + len(key)))
        positions.sort()

        if positions[0] != 0:
            positions.insert(0,0)

        text_split = [row['text'][i:j] for i, j in zip(positions, positions[1:] + [None])]

        tuple_list = []
        for substr in text_split:
            if substr not in row['tagged_words'].keys():
                tmp_tl = []
                if substr.isspace() == False:
                    substr = substr.strip()
                    tmp_sl = substr.split()
                    tmp_tl = [(word.lower(), 'NONE') for word in tmp_sl]
                    tuple_list.extend(tmp_tl)
            else :
                tuple_list.append((substr.lower(),row['tagged_words'][substr]))

        row['split_sentences'] = tuple_list


    # drop annotations and tagged_words column
    columns_to_drop = ['annotations', 'tagged_words']
    data = data.drop(columns_to_drop, axis=1)

    # save the cleaned data into a csv
    data.to_csv(path_or_buf=OUTPUT_PATH, index=False)

    return data




if __name__ == '__main__':
    data = fetch_data(DATA_PATH)
    seed, other = split_data(data)
    seed = transform_labels(seed)
    seed = process_data(seed)

    split_sentences = seed['split_sentences'].values.tolist()
    word_count, unique_words = get_unique_words(split_sentences)
    tags = ['NONE', 'INSTR', 'QLTY']
