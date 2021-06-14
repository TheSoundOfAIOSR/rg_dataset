import pandas as pd
import ast
from collections import Counter

data = pd.read_csv('../reddit_data_preprocessing/data/curated_pattern_lists.csv')

data.pattern.str.count("QLTY").sum()

qualities_list = []
for idx, row in data.iterrows():
    qualities_list += ast.literal_eval(row['QLTY'])
counter = Counter(qualities_list)
df = pd.DataFrame.from_dict(counter, orient='index')

df.to_csv('counter_qualities.csv')