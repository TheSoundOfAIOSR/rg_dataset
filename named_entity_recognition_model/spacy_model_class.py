from __future__ import unicode_literals, print_function
import numpy as np
import pandas as pd
import random
import spacy
from collections import defaultdict
from pathlib import Path
from spacy.util import minibatch, compounding
from reddit_data_preprocessing.data_preprocessing import DATA_PATH, fetch_data, transform_labels


from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator


def split_data(data):
    ''' split the data into labeled (seed) and unlabeled sets

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


def transform_data(data):
    ''' trasnform the labeled dataset into the form ['text', {'entities' : (start_offset, end_offset, label)}]

    :param data: the labeled dataset
    :return: transformed data
    '''
    train_data = []
    for idx, row in data.iterrows():
        entities = {}
        for dic in row['annotations']:
            if 'entities' in entities:
                entities['entities'].append((dic['start_offset'], dic['end_offset'], dic['label']))
            else:
                entities['entities'] = [(dic['start_offset'], dic['end_offset'], dic['label'])]
        train_data.append((row['text'], entities))
    return train_data


class NerModel(BaseEstimator):
    def __init__(self, model=None, n_iter=100, OUTPUT_DIR = Path("./named_entity_recognition_model/model1"), **model_hyper_parameters):
        super().__init__()
        self.model = model
        self.OUTPUT_DIR = OUTPUT_DIR
        self.n_iter = n_iter
        self.classes_ = []

    def fit(self, X, Y=None):
        ''' train the Named Entity Recognition model based

        :param data: the transformed dataset
        :param output_dir: the path where to save the model
        :return: euh... nothing probably
        '''

        data = list(zip(X, Y))
        # dt=np.dtype('int,float')
        data = np.array(data)
        # print(type(data))
        # print(data[50:52])

        if self.model is not None:
            nlp = spacy.load(self.model)
            print("loaded model '%s'" %self.model)
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
        for _, annotations in data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # Initializing optimizer
        if self.model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()

        # Get names of other pipes to disable them during training to train # only NER and update the weights
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(self.n_iter):
                random.shuffle(data)
                losses = {}
                batches = minibatch(data,
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
        if self.OUTPUT_DIR is not None:
            self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        if not self.OUTPUT_DIR.exists():
            self.OUTPUT_DIR.mkdir()
        nlp.to_disk(self.OUTPUT_DIR)
        print("Saved model to", self.OUTPUT_DIR)

    def predict(self, X):
        ''' test the trained model on a random text

        :param text: the tested sentence
        :param output_dir: the path to the model
        :return: again... nothing returned
        '''

        print("Loading from", self.OUTPUT_DIR)
        nlp2 = spacy.load(self.OUTPUT_DIR)
        doc2 = nlp2(X)
        print(X)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

    def predict_proba(self, X):
        # Number of alternate analyses to consider. More is slower, and not necessarily better
        beam_width = 32
        # This clips solutions at each step. We multiply the score of the top-ranked action by this value,
        # and use the result as a threshold. This prevents the parser from exploring options that look very unlikely,
        # saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
        beam_density = 0
        nlp = spacy.load(self.OUTPUT_DIR)


        docs = list(nlp.pipe(X))
        # docs = nlp(X, disable=['ner'])
        beams = nlp.entity.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)

        for doc, beam in zip(docs, beams):
            entity_scores = defaultdict(float)
            for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score


        l= []
        tmp = []
        predicted_probas = []
        self.classes_ = []
        for k, v in entity_scores.items():
            self.classes_.append(k[2])
            # predicted_probas.append(v)
            l.append({'start': k[0], 'end': k[1], 'label': k[2], 'prob' : v})
            if len(tmp) < 2:
                tmp.append(v)
            else :
                predicted_probas.append(tmp)
                tmp = []

        l = sorted(l, key=lambda x: x['start'])

        for a in sorted(l, key= lambda x: (x['start'],x['end'], x['label'])):
            print(a)

        predicted_probas = np.array(predicted_probas)

        return predicted_probas


if __name__ == "__main__":
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
    seed = transform_labels(seed)
    tags = ['NONE', 'INSTR', 'QLTY']

    # transform the labeled data
    train_data = transform_data(seed)
    X, y = [text for text,_ in train_data], [entity for _,entity in train_data]

    X_test = [text for text in other['text']]
    y_test = [dict(entity) for entity in other['annotations']]

    # train the NER model
    ner_model = NerModel()
    ner_model.fit(X, y)


    # test the model
    text1 = ['I used to play guitar, now I play violin and it has some kind of distortion']
    text2 = ["I'd like a sharp cello"]
    # result = ner_model.predict(text2)

    predicted_proba = ner_model.predict_proba(text1)

