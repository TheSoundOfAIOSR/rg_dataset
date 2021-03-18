import pandas as pd
import re
import numpy as np

DATA_PATH = './doccano_data/project_1_dataset.jsonl'
OUTPUT_PATH = './active_learning/processed_data.csv'

def fetch_data(data_path=DATA_PATH):
    ''' fetch the data project_1_dataset.jsonl from the doccano_data repository

    :param data_path: path to the data JSONL file
    :return: dataframe containing all the data with columns (id, text, annotations, meta, annotation_approver, comments)
    '''

    data = pd.read_json(path_or_buf=data_path, lines=True)
    return data


def split_data(data):
    '''

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
    x = re.sub(r'\\n', ' ', x)
    x = re.sub(r'\\u', ' ', x)
    x = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', x)
    x = re.sub('[^-A-Za-z0-9]', ' ', x)
    return ' '.join(x.split())


def process_data(data):
    '''

    :param data:
    :return:
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
    # and the label is either zero (if it's not annotated) or 1-2 if it's annotated
    for idx, row in data.iterrows():
        word_tuples = []
        for word in row['text'].split():
            if word not in row['tagged_words'].keys() :
                word_tuples.append((word,0))
            else:
                word_tuples.append((word, row['tagged_words'][word]))
        row['split_sentences'] = word_tuples

    # drop annotations and tagged_words column
    columns_to_drop = ['annotations', 'tagged_words']
    data = data.drop(columns_to_drop, axis=1)

    # save the cleaned data into a csv
    data.to_csv(path_or_buf=OUTPUT_PATH, index=False)

    return data




if __name__ == '__main__':
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
    seed = process_data(seed)

