import pandas as pd
import re

DATA_PATH = './doccano_data/project_1_dataset.jsonl'

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

    # extract tuples (word, label) from the annotation column
    # and add them to the tagged words column
    for idx, row in data.iterrows():
        tuple_list = []
        for dict in row['annotations']:
            tuple_list.append((row['text'][dict['start_offset'] : dict['end_offset']], dict['label']))
        row['tagged_words'] = tuple_list


    # clean text data
    data['text'] = data['text'].apply(remove_special_chars)

    # split text into a list of tuples (word, label) where the word is the word in each text
    # and the label is either zero (if it's not annotated) or 1-2 if it's annotated


    return data




if __name__ == '__main__':
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
    seed = process_data(seed)

