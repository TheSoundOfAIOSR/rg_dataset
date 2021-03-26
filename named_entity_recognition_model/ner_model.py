import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from named_entity_recognition_model.data_tokenization import tokenize_element
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, concatenate, \
                         SpatialDropout1D

MODEL_PATH = "ner_model.h5"

def train_model(model_path=MODEL_PATH):

    tags = ['INSTR', 'QLTY', 'NONE']
    n_tags = len(tags)

    # load the sequences of words, characters and annotations
    X_word, X_char, y, max_len, max_len_char, n_words, n_chars, idx2word, idx2tag = tokenize_element()
    X_char = np.asarray(X_char).astype(np.float32)

    # split the data into train and test datasets
    X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=42)
    X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=42)

    print(type(X_word_tr))
    print(type(X_char_tr))

    # X_char_tr = np.asarray(X_char_tr)
    # X_char_te = np.asarray(X_char_te)

    # imput and embedding for words
    word_in = Input(shape=(max_len,))
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                         input_length=max_len, mask_zero=True)(word_in)

    # input and embeddings for characters
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                         input_length=max_len_char, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                    recurrent_dropout=0.5))(emb_char)

    # main LSTM
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)

    model = Model([word_in, char_in], out)

    # compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.summary()

    # train phase
    history = model.fit([X_word_tr,
                         np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                        np.array(y_tr).reshape(len(y_tr), max_len, 1),
                        batch_size=32, epochs=10, validation_split=0.1, verbose=1)
    hist = pd.DataFrame(history.history)

    #plot the histograms
    plt.style.use("ggplot")
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()

    model.save(model_path)

    return model, X_word_te, X_char_te, y_te, max_len, max_len_char, idx2word, idx2tag

def predict_results(model, X_word_te, X_char_te, y_te, max_len, max_len_char, idx2word, idx2tag):

    y_pred = model.predict([X_word_te,
                            np.array(X_char_te).reshape((len(X_char_te),
                                                         max_len, max_len_char))])

    i = 2
    p = np.argmax(y_pred[i], axis=-1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_word_te[i], y_te[i], p):
        if w != 0:
            print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))

if __name__ == "__main__":
    model, X_word_te, X_char_te, y_te, max_len, max_len_char, idx2word, idx2tag = train_model()
    predict_results(model, X_word_te, X_char_te, y_te, max_len, max_len_char, idx2word, idx2tag)

