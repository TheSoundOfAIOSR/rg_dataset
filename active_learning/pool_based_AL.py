from progress.spinner import MoonSpinner
import sys
from pyfiglet import Figlet
from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint

import numpy as np
import pandas as pd
import time
from modAL.models import ActiveLearner
from named_entity_recognition_model.spacy_model_class import NerModel, split_data, transform_data
from reddit_data_preprocessing.data_preprocessing import fetch_data, DATA_PATH, transform_labels


# set the Style for some elements of our CLI
style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '#673ab7 bold',
})


# Set the RNG seed for reproducibility
RANDOM_STATE_SEED = 42
np.random.seed(RANDOM_STATE_SEED)


class ActiveLearning():
    def __init__(self, data = fetch_data(data_path=DATA_PATH)):
        self.data = data


    def split_new_data(self, data):
        ''' split the data into a labeled set (seed) and an unlabeled set

        :param data: data containing labeled and unlabeled data
        :return: dataframe called seed containing all the labeled data and unlabelled_dataset containing unlabelled data
        '''
        column_names = data.columns
        important_columns = ['text', 'labels']

        seed = pd.DataFrame(columns=important_columns)
        unlabelled_dataset = pd.DataFrame(columns=important_columns)

        for idx, row in data.iterrows():
            if not row['labels'] :
                unlabelled_dataset = unlabelled_dataset.append(row[important_columns], ignore_index=True)
            else :
                seed = seed.append(row[important_columns], ignore_index=True)

        return seed, unlabelled_dataset


    def transform_new_data(self, data):
        ''' trasnform the labeled dataset into the form ['text', {'entities' : (start_offset, end_offset, label)}]

        :param data: the labeled dataset
        :return: transformed data
        '''
        train_data = []
        for idx, row in data.iterrows():
            entities = {}
            for tag in row['labels']:
                if 'entities' in entities:
                    entities['entities'].append(tuple(tag))
                else:
                    entities['entities'] = [tuple(tag)]
            train_data.append((row['text'], entities))
        return train_data


    def load_split(self, data):
        '''

        :param data:
        :return:
        '''

        seed, other = split_data(data)
        seed = transform_labels(seed)

        # transform the labeled data to be in Spacy format
        train_data = transform_data(seed)
        # split the labeled data into features and labels
        X_train, y_train = [text for text,_ in train_data], [entity for _,entity in train_data]

        # split the unlabeled data
        X_pool = [text for text in other['text']]
        y_pool = [dict(entity) for entity in other['annotations']]

        return X_train, X_pool, y_train, y_pool


    def query_sentences(self):
        ''' fetch for the most informative sentences from the unlabelled dataset
            and waits for the oracle(user) to annotate them

        :yield: queried sentences and indexes each time the model is updated
        '''

        f = Figlet(font='doom', width=80)
        print(f.renderText('Active Learning'))

        start_time = time.time()
        print(' Loading data ...')
        print(' Splitting and transforming data ...')
        X_train, X_pool, y_train, y_pool = self.load_split(self.data)
        print(' Successfully preprocessed the data !')

        # specify our core estimator along with it's active learning model
        ner_model = NerModel()
        print(' Collecting estimator and training data : Done !')
        learner = ActiveLearner(estimator=ner_model, X_training=X_train, y_training=y_train)

        print("\n To start the annotation process using active Learning you need to specify :", end='\n       |_ ')
        print("Nb of queries (How many times you want to update the NER model by sampling the unlabelled data)",
              end='\n       |_ ')
        print("Nb of instances (How many informative sentences should be returned)")

        # Update our model by pool-based sampling our "unlabeled" dataset
        questions = [
            {
                'type': 'input',
                'name': 'n_queries',
                'message': 'Number of queries : '
            },
            {
                'type': 'input',
                'name': 'n_instances',
                'message': 'Number of instances : '
            }
        ]
        answers = prompt(questions, style=style)
        N_QUERIES = int(answers['n_queries'])

        # Allow our model to query our unlabeled dataset for the most
        # informative points according to our query strategy (uncertainty sampling)
        for idx in range(N_QUERIES):
            print(" Query n°", idx + 1)
            print(" Sampling sentences...")
            query_idx, query_instance = learner.query(X_pool, n_instances=int(answers['n_instances']))
            print(" Elapsed time : %s seconds" % (time.time() - start_time))
            # yield query_idx, query_instance
            # print(query_idx)
            pprint(query_instance)

            sd_questions = [
                {
                    'type': 'input',
                    'name': 'path',
                    'message': 'Please provide a path to the newly annotated data : '
                },
                {
                    'type': 'input',
                    'name': 'confirmation',
                    'message': 'Proceed : (y/N) '
                }
            ]
            sd_answers = prompt(sd_questions, style=style)

            if sd_answers['confirmation'].lower() == 'y':
                # fetch and the new annotated data
                new_data = fetch_data(data_path=sd_answers['path'])
                new_seed, _ = self.split_new_data(new_data)
                new_seed = self.transform_new_data(new_seed)

                X = []
                y=[]
                for sentence in query_instance:
                    for seed in new_seed:
                        if seed[0]==sentence :
                            X.append(sentence)
                            y.append(seed[1])

                X = np.array(X)
                y = np.array(y)

                #Teach our ActiveLearner model the record it has requested
                print(' Teaching the NER model the newly annotated data...')
                learner.teach(X=X, y=y)

                # remove the queried instance from the unlabeled pool
                sorted_query_idx = sorted(query_idx, reverse=True)
                for index in sorted_query_idx:
                    del X_pool[index]
                print(' Removing queried data from the pool : Done')

            else :
                break


if __name__ == "__main__":
    al = ActiveLearning()
    al.query_sentences()
    # for _, sentences in al.query_sentences():
    #     print(sentences)

