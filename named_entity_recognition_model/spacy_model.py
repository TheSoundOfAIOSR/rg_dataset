from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pandas as pd
from reddit_data_preprocessing.data_preprocessing import DATA_PATH, fetch_data, split_data, get_unique_words,process_data, transform_labels
import random
from __future__ import unicode_literals, print_function
import plac
from tqdm import tqdm


#nlp = spacy.load("en_core_web_sm")

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

train_data = []
for idx, row in seed.iterrows():
    entities = {}
    for dic in row['annotations']:
        if 'entities' in entities:
            entities['entities'].append((dic['start_offset'], dic['end_offset'], dic['label']))
        else:
            entities['entities'] = [(dic['start_offset'], dic['end_offset'], dic['label'])]
    train_data.append((row['text'], entities))


model = None
output_dir=Path("./named_entity_recognition_model/model1")
n_iter = 100

if model is not None:
    nlp = spacy.load(model)
    print("loaded model '%s'" %model)
else :
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe('ner')

# add labels, Trains data based on annotations
for _, annotations in train_data:
    for ent in annotations.get('entities'):
        print(ent[2])
        ner.add_label(ent[2])

# Initializing optimizer
if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()

# get names of other pipes to disable them during training
# other_pipes = [pipe for pipe in ner_model.pipe_names if pipe != 'ner']
# with ner_model.disable_pipes(*other_pipes):  # only train NER
#     optimizer = ner_model.begin_training()
#     for itn in range(n_iter):
#         random.shuffle(train_data)
#         losses = {}
#         for text, annotations in tqdm(train_data):
#             ner_model.update(
#                 [text],  # batch of texts
#                 [annotations],  # batch of annotations
#                 drop=0.5,  # dropout
#                 sgd=optimizer,  # callable to update weights
#                 losses=losses)
#         print(losses)

# Get names of other pipes to disable them during training to train # only NER and update the weights
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data,
                            size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            # Updating the weights
            nlp.update(texts, annotations, sgd=optimizer,
                       drop=0.35, losses=losses)
        print('Losses', losses)
        nlp.update(texts, annotations, sgd=optimizer,
                   drop=0.35, losses=losses)
    print('Losses', losses)

# save model to output directory
if output_dir is not None:
    output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir()
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
text = 'I used to play guitar, my guitar had some kind of distortion'
text2 = "I'd like a sharp cello"
doc2 = nlp2(text2)
print(text2)
for ent in doc2.ents:
    print(ent.label_, ent.text)