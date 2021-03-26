from pathlib import Path
import spacy
import pandas as pd
from reddit_data_preprocessing.data_preprocessing import DATA_PATH, fetch_data, split_data, get_unique_words,process_data, transform_labels

def transform_data(data):
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

data = fetch_data(data_path=DATA_PATH)
seed, other = transform_data(data)
seed = transform_labels(seed)
# seed = process_data(seed)
# word_count, unique_words = get_unique_words(seed['text'])

#split_sentences = seed['split_sentences'].values.tolist()
tags = ['NONE', 'INSTR', 'QLTY']


model = None
ouput_dir=Path("./active_learning/model1")
n_iter = 100

if model is not None:
    ner_model = spacy.load(model)
    print("loaded model '%s'" %model)
else :
    ner_model = spacy.blank('en')
    print("Created blank 'en' model")

train_data = []
for idx, row in seed.iterrows():
    entities = {}
    for dic in row['annotations']:
        if 'entities' in entities:
            entities['entities'].append((dic['start_offset'], dic['end_offset'], dic['label']))
        else:
            entities['entities'] = [(dic['start_offset'], dic['end_offset'], dic['label'])]
    train_data.append((row['text'], entities))