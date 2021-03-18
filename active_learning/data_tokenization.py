from keras.preprocessing.sequence import pad_sequences
from active_learning.data_preprocessing import DATA_PATH, OUTPUT_PATH, fetch_data, split_data, remove_special_chars,\
                                               get_unique_words,process_data


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

    # we have 3 tags {0: non-annotated, 1: INSTRUMENT, 2: QUALITY/TIMBRE
    tags = ['NONE', 'INSTR', 'QLTY']

    # load and preprocess the data
    data = fetch_data(data_path=DATA_PATH)
    seed, other = split_data(data)
    seed = process_data(seed)
    word_count, unique_words = get_unique_words(seed['text'])
    unique_words = list(unique_words)

    # transforming the split_sentences into a list of lists
    split_sentences = seed['split_sentences'].values.tolist()
    print(split_sentences)

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

    print(word2idx["guitar"])
    print(tag2idx["INSTR"])

    # map the sentences to a sequence of numbers and pad the sequence
    X_word = [[word2idx[w[0]] for w in s] for s in split_sentences]
    X_word = pad_sequences(max_len=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

    # generate dictionary for the characters and create the sequence of characters for every token
    chars = set([w_i for w in unique_words for w_i in w])
    n_chars = len(chars)
    print(n_chars)




if __name__ == '__main__':
    tokenize_element()