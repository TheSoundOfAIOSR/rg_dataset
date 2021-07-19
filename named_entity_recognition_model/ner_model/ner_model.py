import numpy
import random
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
    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    return ner, nlp


class NerModel(BaseEstimator):
    def __init__(self, ner, nlp, n_iter=64, dropout=0.1, lr=0.001, **model_hyper_parameters):
        super().__init__()
        self.ner = ner
        self.nlp = nlp
        self.n_iter = n_iter
        self.dropout = dropout
        self.lr = lr

    def clear_model(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.ner = self.nlp.get_pipe("ner")

    def fit(self, train_data, eval_data):
        """ train the Named Entity Recognition model

        :param eval_data: evaluation data for testing after every epoch
        :param train_data: processed training data
        :return: evaluation fscore of the final epoch
        """
        # Adding labels to the NER
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Disable pipeline components that are not changed
        pipe_exceptions = ["ner"]
        unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        scorer = Scorer()

        # Store the PRF scores for every iteration
        train_scores = []
        eval_scores = []

        # Store losses after every iteration
        # Each loss is itself an average of losses within a single iteration
        loss_list = []

        # Train the NER model
        with self.nlp.select_pipes(enable=pipe_exceptions, disable=unaffected_pipes):
            # Create a list of Examples objects
            examples = []

            for text, annots in train_data:
                examples.append(Example.from_dict(self.nlp.make_doc(text), annots))

            # Create an optimizer for the pipeline component, and set lr
            optimizer = self.nlp.create_optimizer()

            # optimizer = nlp.initialize()
            # NOTE: Cannot use nlp.initilaize (v3) (aka nlp.begin_training for v2) on pretrained models.
            # Use nlp.create_optimizer for training on existing model (We used pretrained en_core_web_sm).
            # ref: https://stackoverflow.com/a/66369163/6475377

            optimizer.learn_rate = self.lr

            for iteration in range(ITERATIONS):
                # print("Iteration: ", iteration)
                # shuffling examples  before every iteration
                random.shuffle(examples)
                losses = {}

                # optimizer = self.nlp.resume_training()

                # batch up the examples using spaCy's minibatch
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for count, batch in enumerate(batches):
                    self.nlp.update(
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
                    pred_value = self.nlp(text)
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

            self.nlp.to_disk("./saved_model")

        draw_prf_graph(train_scores, keyword="train")
        draw_prf_graph(eval_scores, keyword="eval")
        draw_train_eval_compare_graph(train_scores, eval_scores)
        plot_training_loss_graph(loss_list, "Losses with epochs")

        # Just write the last epoch's eval fscore in txt file
        eval_fscore = []
        for i, eval_score in enumerate(eval_scores):
            for key, cat in eval_score.items():
                if key == "ents_f": eval_fscore.append(cat)

        # with open("img/k_cv_scores.txt", 'a') as f:
        #     f.write("%s\n" % str(eval_fscore[-1]))

        return eval_fscore[-1]

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
            doc_gold_text = self.nlp.make_doc(text)

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

    def k_cross_validation(self, data, k=10):
        print(f"{k}-fold Cross Validation")
        random.shuffle(data)
        num_groups = int(len(data) / k)
        print(f"Size of each eval set: {num_groups}\n")
        batches = minibatch(data, size=num_groups)

        for count, batch in enumerate(batches):
            # Discard the last batch if it has very few example sentences
            if len(batch) > num_groups / 2:
                print(f"Fold no.: {count + 1}")
                train_data = [x for x in data if x not in batch]
                test_data = batch
                print(f"Train, Test :: {len(train_data)}, {len(test_data)}")
                fscore = self.fit(train_data=train_data, eval_data=test_data)
                print(f"fscore: {fscore}\n")

            self.clear_model()


if __name__ == '__main__':
    print("spaCy version: ", spacy.__version__)
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

    # Create the NER model class consisting of fit and evaluate methods.
    ner_model = NerModel(ner, nlp, n_iter=ITERATIONS, dropout=DROPOUT, lr=LEARN_RATE)

    # We're gonna use TEST (5% + 5% = 10%) for evaluation
    TEST = EVAL_DATA + TEST_DATA
    print("Size of total TRAIN data: ", len(TRAIN_DATA))
    print("Size of total TEST (Evaluation) data: ", len(TEST))
    ner_model.fit(TRAIN_DATA, TEST)

    # Perform k-fold Cross Validation
    # data = TRAIN_DATA + EVAL_DATA + TEST_DATA
    # ner_model.k_cross_validation(data, k=10)

    # sentence = 'I really like the distortion in this guitar'
    # ner.predict(sentence)
