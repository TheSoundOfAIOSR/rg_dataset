import ast

import numpy as np
import pandas as pd
import time
from modAL.models import ActiveLearner
from named_entity_recognition_model.spacy_model_class import NerModel, split_data, transform_data
from reddit_data_preprocessing.data_preprocessing import fetch_data, DATA_PATH, transform_labels

# Set the RNG seed for reproducibility
RANDOM_STATE_SEED = 42
np.random.seed(RANDOM_STATE_SEED)

data = fetch_data(data_path=DATA_PATH)


def split_new_data(data):
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


def transform_new_data(data):
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

def load_split(data):
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

start_time = time.time()
X_train, X_pool, y_train, y_pool = load_split(data)
# specify our core estimator along with it's active learning model
ner_model = NerModel()
learner = ActiveLearner(estimator=ner_model, X_training=X_train, y_training=y_train)

# Update our model by pool-based sampling our "unlabeled" dataset
N_QUERIES = 20

# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling)
for idx in range(N_QUERIES):
    query_idx, query_instance = learner.query(X_pool, n_instances=10)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(query_idx)
    print(query_instance)

    print("##########  ##########")
    print("Query n°", idx)
    confirmation = input("Continue ? (y/n) :")

    if confirmation.lower() == 'y':
        # fetch and the new annotated data
        new_data = fetch_data(data_path='./doccano_data/project_2_dataset.jsonl')
        new_seed, _ = split_new_data(new_data)
        new_seed = transform_new_data(new_seed)

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
        learner.teach(X=X, y=y)

        # remove the queried instance from the unlabeled pool
        sorted_query_idx = sorted(query_idx, reverse=True)
        for index in sorted_query_idx:
            del X_pool[index]

    else :
        break



    # X= [X_pool[query_idx[0]]]
    #
    # # ask the user to label the sentence
    # print("The sentence that needs to be labeled is :")
    # print(X)
    # print("Which words should be labeled ?")
    # labels = input("Format (start_offset, end_offset, 'label') : ")
    #
    # y = np.array([{'entities':[ast.literal_eval(x) for x in labels.splitlines()]}])
    # X = np.array(X)
    #
    # #Teach our ActiveLearner model the record it has requested
    # learner.teach(X=X, y=y)
    #
    # # remove the queried instance from the unlabeled pool
    # X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)




# Let's see how our classifier performs on the initial training set
# predictions = learner.predict(X_train)
# is_correct = (predictions == y_train['entities'])




# if __name__ == "__main__":
#     data = fetch_data(data_path=DATA_PATH)