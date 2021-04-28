import re
import pandas as pd
import matplotlib.pyplot as plt
import operator
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

FILE_PATH = './reddit_data_preprocessing/processed_data.csv'

def fetch_data(path=FILE_PATH):
    data = pd.read_csv(path)
    return data

data = fetch_data()

stop_words = set(stopwords.words('english'))

def remove_stopwords(sentence, stop_words=stop_words):
    tokenized_sentence = word_tokenize(sentence)
    filtered_sentence = ' '.join([word for word in tokenized_sentence if word not in stop_words])
    return filtered_sentence

data['text'] = data['text'].apply(remove_stopwords)

plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800).generate(" ".join(data.text))
plt.imshow(wc, interpolation = 'bilinear')