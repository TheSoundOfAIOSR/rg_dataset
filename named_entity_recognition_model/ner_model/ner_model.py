import numpy
import pandas as pd
import random
from tqdm.auto import tqdm
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
from sklearn.base import BaseEstimator
from utilities import load_cleaned_data, split_data

numpy.random.seed(0)


def load_spacy():
    nlp = spacy.load("en_core_web_sm")
    spacy.__version__
    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    return ner, nlp


class NerModel(BaseEstimator):
    def __init__(self, ner,  nlp, model=None, n_iter=64, dropout=0.1,  **model_hyper_parameters):
        super().__init__()
        self.ner = ner
        self.nlp = nlp
        self.model = model
        self.n_iter = n_iter
        self.dropout = dropout

    def fit(self, train_data):
        ''' train the Named Entity Recognition model

        :param train_data: processed training data
        :return: None
        '''
        # Adding labels to the NER
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Disable pipeline components that are not changed
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        # Train the NER model
        with self.nlp.disable_pipes(*unaffected_pipes):
            for iteration in tqdm(range(self.n_iter),
                                  desc=" Training the NER model on annotated data"):
                # print("Iteration: ", iteration)
                # shuufling examples  before every iteration
                random.shuffle(train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=self.dropout,  # dropout - make it harder to memorise data
                        losses=losses
                    )
                    # print("Losses", losses)

        self.nlp.to_disk("./saved_model")

    def evaluate(self, test_data):
        ''' test the trained NER model

        :param test_data: processed test data
        :return: None
        '''
        for example in test_data:
            print(example[0])
        doc = self.nlp(example[0])
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    def predict(self, X):
        ''' make inferences on unseen data

        :param X: sentence to make inferences on
        :return: None
        '''
        self.nlp = spacy.load("./saved_model")
        doc = self.nlp(X)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    # def predict_proba(self):
    #     '''
    #
    #     :return:
    #     '''

if __name__ == '__main__':

    ner, nlp = load_spacy()
    # DATA = load_cleaned_data()
    # TRAIN_DATA, TEST_DATA = split_data(DATA)
    ner = NerModel(ner, nlp)
    # ner.fit(TRAIN_DATA)
    # ner.evaluate(TEST_DATA)

    sentence = 'I really like the distortion in this guitar'
    ner.predict(sentence)



