import numpy as np
from keras.preprocessing.sequence import pad_sequences
from reddit_data_preprocessing.data_preprocessing import DATA_PATH, fetch_data, split_data, get_unique_words,process_data, transform_labels


def max_length(list):
    ''' finds the maximum length of elements in a list

    :param list: can be the list of sentences or list of words
    :return: max length of the elements of a list
    '''

    length = 0

    for element in list:
        if length < len(element):
            length = len(element)

    return length


def tokenize_element():

    # we have 3 tags {1: INSTRUMENT, 2: QUALITY/TIMBRE, 3: non-annotated}
    tags = ['INSTR', 'QLTY', 'NONE']

    # load and preprocess the data
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
    seed = transform_labels(seed)
    seed = process_data(seed)
    word_count, unique_words = get_unique_words(seed['text'])
    unique_words = list(unique_words)
    n_words = len(unique_words)

    # transforming the split_sentences into a list of lists
    split_sentences = seed['split_sentences'].values.tolist()
    # print(split_sentences)

    # get lengths of longest sentence and longest words
    max_len = max_length(seed['split_sentences'])
    max_len_char = max_length(unique_words)

    # creating dictionaries of words and tags
    word2idx = {w: i + 2 for i, w in enumerate(unique_words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    # print(word2idx["guitar"])
    # print(tag2idx["INSTR"])

    # map the sentences to a sequence of numbers and pad the sequence
    X_word = [[word2idx[w[0]] for w in s] for s in split_sentences]
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

    # generate dictionary for the characters and create the sequence of characters for every token
    chars = set([w_i for w in unique_words for w_i in w])
    n_chars = len(chars)
    print(n_chars)

    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    X_char = []
    for sentence in split_sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(split_sentences[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    y = [[tag2idx[w[1]] for w in s] for s in split_sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

    return X_word, X_char, y, max_len, max_len_char, n_words, n_chars, idx2word, idx2tag



if __name__ == '__main__':
    X_word, X_char, y, max_len, max_len_char, n_words, n_chars, idx2word, idx2tag = tokenize_element()