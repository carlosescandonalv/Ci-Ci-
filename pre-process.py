import nltk
import re
import string
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sn
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import emoji
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

df = pd.read_csv('tweets_scrapped.csv')

STOP_WORDS = nltk.corpus.stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()

def remove_html_tags(text):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))


def remove_pictureURL(text):
    return re.sub('pic?.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', "", text)


def remove_space_mentions(text):
    split_word = '@ '
    for i in text:
        if i == '@':
            res_str = text.partition(split_word)[2]
            final_sentence = res_str.split(' ', 1)[1:]
        else:
            final_sentence = text
        return final_sentence


def remove_complex_mentions(text):
    exceptions = '_[A-Z]'
    for i in text:
        return "".join([i for i in text if i not in exceptions])


def remove_mentions(text):
    return re.sub('@[A-Za-z0-9]+', '', str(text))


def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])


def remove_emoji(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def tokenize(text):
    return nltk.word_tokenize(text)


# eliminate unimportant words
def remove_stopwords(text):
    return [i for i in text if i not in STOP_WORDS]


# switches a word to its base root mode
def lemmatize(text):
    pos_tagged = nltk.pos_tag(text)
    lemmatized_words = []

    for word, tag in pos_tagged:
        tag = tag[0].lower()

        if tag not in ['a', 'r', 'n', 'v']:
            lemma = word
        else:
            lemma = LEMMATIZER.lemmatize(word, tag)

        lemmatized_words.append(lemma)

    return " ".join(lemmatized_words)


def remove_quotation_marks(text):
    return "".join([i for i in text if i != "’"])


def pre_process(text):
    text = remove_html_tags(text)
    text = remove_pictureURL(text)
    text = remove_space_mentions(text)
    text = remove_complex_mentions(text)
    text = remove_mentions(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_emoji(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = remove_quotation_marks(text)
    return text

df["pre_processed"] = df["Tweet"].apply(lambda x: pre_process(x))
df.drop_duplicates(["pre_processed"], inplace=True)
df['pre_processed'].replace('', np.nan, inplace=True)
df.dropna(inplace=True)

df[['pre_processed']].to_csv('scrapped_pre_processed.csv')
