import pandas as pd

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

    seed = pd.DataFrame(columns=column_names)
    unlabelled_dataset = pd.DataFrame(columns=column_names)

    for idx, row in data.iterrows():
        if not row['annotations'] :
            unlabelled_dataset = unlabelled_dataset.append(row, ignore_index=True)
        else :
            seed = seed.append(row, ignore_index=True)

    return seed, unlabelled_dataset


if __name__ == '__main__':
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
