import numpy
import pandas as pd
import random
from tqdm.auto import tqdm
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.scorer import Scorer
from sklearn.base import BaseEstimator
from utilities import load_cleaned_data, split_data, DROPOUT, ITERATIONS, draw_prf_graph, plot_training_loss_graph, \
    draw_train_eval_compare_graph, save_list_to_pickle, load_list_from_pickle, LEARN_RATE
import pickle

numpy.random.seed(0)


def load_spacy():
    nlp = spacy.load("en_core_web_sm")
    print("spaCy version: ", spacy.__version__)
    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    return ner, nlp


class NerModel(BaseEstimator):
    def __init__(self, ner, nlp, model=None, n_iter=64, dropout=0.1, **model_hyper_parameters):
        super().__init__()
        self.ner = ner
        self.nlp = nlp
        self.model = model
        self.n_iter = n_iter
        self.dropout = dropout

    def fit(self, train_data, eval_data):
        """ train the Named Entity Recognition model

        :param eval_data: evaluation data for testing after every epoch
        :param train_data: processed training data
        :return: None
        """
        # Adding labels to the NER
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Disable pipeline components that are not changed
        pipe_exceptions = ["ner"]
        unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        scorer = Scorer()

        # Store the PRF scores for every iteration
        train_scores = []
        eval_scores = []

        # Store losses after every iteration
        # Each loss is itself an average of losses within a single iteration
        loss_list = []

        # Train the NER model
        with nlp.select_pipes(enable=pipe_exceptions, disable=unaffected_pipes):
            # Create a list of Examples objects
            examples = []

            for text, annots in train_data:
                examples.append(Example.from_dict(nlp.make_doc(text), annots))

            # Create an optimizer for the pipeline component, and set lr
            optimizer = nlp.create_optimizer()

            # optimizer = nlp.initialize()
            # NOTE: Cannot use nlp.initilaize (v3) (aka nlp.begin_training for v2) on pretrained models.
            # Use nlp.create_optimizer for training on existing model (We used pretrained en_core_web_sm).
            # ref: https://stackoverflow.com/a/66369163/6475377

            optimizer.learn_rate = LEARN_RATE

            for iteration in range(ITERATIONS):
                print("Iteration: ", iteration)
                # shuffling examples  before every iteration
                random.shuffle(examples)
                losses = {}

                # optimizer = nlp.resume_training()

                # batch up the examples using spaCy's minibatch
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for count, batch in enumerate(batches):
                    nlp.update(
                        batch,
                        drop=DROPOUT,  # dropout - make it harder to memorise data
                        losses=losses,
                        sgd=optimizer
                    )

                loss = losses["ner"] / (count + 1)
                print(f"Loss at epoch {iteration}: ", loss)
                loss_list.append(loss)
                # After training every iteration, calculate scores
                example_list = []
                for text, annot in train_data:
                    # Create a Doc of our text
                    # doc_gold_text = nlp.make_doc(text)
                    pred_value = nlp(text)
                    # reference = (Example.from_dict(doc_gold_text, annot))
                    gold_standard = {"entities": annot["entities"]}

                    # Store prediction and gold standard ref. for each sentence
                    # (to be used by Scorer.score)
                    example_list.append(Example.from_dict(pred_value, gold_standard))

                # Generate per-entity scores by comparing predicted with gold-standard values
                scores = scorer.score(examples=example_list)
                train_scores.append(scores)

                # Evaluate on eval_data
                eval_scores.append(self.evaluate(test_data=eval_data))

        draw_prf_graph(train_scores, keyword="train")
        draw_prf_graph(eval_scores, keyword="eval")
        draw_train_eval_compare_graph(train_scores, eval_scores)

        plot_training_loss_graph(loss_list, "Losses with epochs")
        self.nlp.to_disk("./saved_model")

    def evaluate(self, test_data):
        """ evaluate the trained NER model

        :param test_data: processed test data
        :return: None
        """
        # for example in test_data:
        #     print(example[0])
        #     doc = self.nlp(example[0])
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

        scorer = Scorer(self.nlp)
        example_list = []

        random.shuffle(test_data)

        # Get the PRF scores for test_data
        for text, annot in test_data:
            # Create a Doc of our text
            doc_gold_text = nlp.make_doc(text)

            # Create gold-standard using the Doc of text
            # and original (correct) entities
            gold_standard = {"text": doc_gold_text, "entities": annot["entities"]}

            # Get the predictions of current test data sentence
            pred_value = self.nlp(text)

            # Create and append to the example list (of type Example) the prediction
            # as well as the gold standard (reference)
            example_list.append(Example.from_dict(pred_value, gold_standard))

        # Generate per-entity scores by comparing predicted with gold-standard values
        scores = scorer.score(examples=example_list)

        # print("All scores: ", scores)
        #
        # print("\nents_p (aka Precision): ", scores['ents_p'])
        # print("ents_r (aka Recall): ", scores['ents_r'])
        # print("ents_f (aka fscore): ", scores['ents_f'])
        #
        # print("\nINSTR: ", scores['ents_per_type']['INSTR'])
        # print("QLTY: ", scores['ents_per_type']['QLTY'])
        # print("EDGE: ", scores['ents_per_type']['EDGE'])
        # print("\n")

        return scores

    def test(self, test_data):
        """
        Perform final testing on unseen test_data
        :param test_data: the unseen test data
        :return:
        """
        # TODO

    def predict(self, X):
        """ make inferences on unseen data

        :param X: sentence to make inferences on
        :return: None
        """
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
    # TRAIN_DATA, EVAL_DATA, TEST_DATA = split_data(DATA)
    # save_list_to_pickle(TRAIN_DATA, "TRAIN_DATA")
    # save_list_to_pickle(EVAL_DATA, "EVAL_DATA")
    # save_list_to_pickle(TEST_DATA, "TEST_DATA")

    # Load pickled data list from data folder
    TRAIN_DATA = load_list_from_pickle("TRAIN_DATA")
    EVAL_DATA = load_list_from_pickle("EVAL_DATA")
    TEST_DATA = load_list_from_pickle("TEST_DATA")

    # We're gonna use TEST (5% + 5% = 10%) for evaluation
    TEST = EVAL_DATA + TEST_DATA
    print("Size of total TEST data: ", len(TEST))
    ner = NerModel(ner, nlp, n_iter=ITERATIONS, dropout=DROPOUT)
    ner.fit(TRAIN_DATA, TEST)

    sentence = 'I really like the distortion in this guitar'
    ner.predict(sentence)
