import pandas as pd
from ast import literal_eval

DATA_PATH = './reddit_data_preprocessing/data/curated_data.csv'
OUTPUT_PATH = './reddit_data_preprocessing/data/curated_pattern_lists.csv'


def load_data(data_path=DATA_PATH):
    loaded_data = pd.read_csv(data_path, sep=';', converters={"split_sentences": literal_eval})
    return loaded_data


def extract_patterns(sentence):
    '''
    :param sentence: list of tuples
    :return: pattern sentence without the actual instruments and qualities. E.g. I heard a QLTY INSTR
             list of instruments
             list of qualities
             list of edge words
    '''

    pattern = ''
    instruments = []
    qualities = []

    # extract patterns and append them to lists
    for word, tag in sentence:
        if tag == 'NONE':
            pattern = pattern + f" {word}"
        elif tag == 'INSTR':
            pattern = pattern + f" {tag}"
            instruments.append(word)
        elif tag == 'QLTY':
            pattern = pattern + f" {tag}"
            qualities.append(word)

    return pattern, instruments, qualities


def process_data(data, output_path=OUTPUT_PATH):

    # create a dataframe to store the results
    column_names = ['pattern', 'INSTR', 'QLTY']
    patterns = pd.DataFrame(columns=column_names)

    # apply the extract pattern to the column
    for idx, row in data.iterrows():
        pattern, instruments, qualities = extract_patterns(row['split_sentences'])
        result_row = {'pattern': pattern,
                      'INSTR': instruments,
                      'QLTY': qualities}
        patterns = patterns.append(result_row, ignore_index=True)

    # save the dataframe into a csv
    patterns.to_csv(path_or_buf=output_path, index=False)


if __name__ == '__main__':
    data = load_data()
    process_data(data, OUTPUT_PATH)
