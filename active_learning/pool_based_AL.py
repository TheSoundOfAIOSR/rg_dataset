import numpy as np
from modAL.models import ActiveLearner
from named_entity_recognition_model.spacy_model_class import NerModel, split_data, transform_data
from reddit_data_preprocessing.data_preprocessing import fetch_data, DATA_PATH, transform_labels

# Set the RNG seed for reproducibility
RANDOM_STATE_SEED = 42
np.random.seed(RANDOM_STATE_SEED)

data = fetch_data(data_path=DATA_PATH)
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


X_train, X_pool, y_train, y_pool = load_split(data)
# specify our core estimator along with it's active learning model
ner_model = NerModel()
learner = ActiveLearner(estimator=ner_model, X_training=X_train, y_training=y_train)

# Update our model by pool-based sampling our "unlabeled" dataset
N_QUERIES = 20

# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling)
for idx in range(N_QUERIES):
    query_idx, query_instance = learner.query(X_pool, n_instances = 5)

    #Teach our ActiveLearner model the record it has requested
    X, y = X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, )
    learner.teach(X=X, y=y)

    # remove the queried instance from the unlabeled pool
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)




# Let's see how our classifier performs on the initial training set
# predictions = learner.predict(X_train)
# is_correct = (predictions == y_train['entities'])




# if __name__ == "__main__":
#     data = fetch_data(data_path=DATA_PATH)