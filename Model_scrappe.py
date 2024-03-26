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
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import plot_confusion_matrix as plot_conf


df = pd.read_csv('train_pre_processed.csv')

def view_distribution(data, subset):
    plt.figure(figsize=(6,5))
    plt.title("Number of different sentiments - "+subset)
    plot = sn.countplot(x = data['sentiment'], data=data)
    for p in plot.patches:
        plot.annotate(p.get_height(),(p.get_x()+0.1 ,p.get_height()+50))

view_distribution(df, "pre processed")

count_vec = CountVectorizer()

X = df["pre_processed"] #input
y = df["sentiment"] #output

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state = 42)

X_train = count_vec.fit_transform(X_train)
X_test = count_vec.transform(X_test)

#model = MultinomialNB().fit(X_train, y_train)
model = svm.SVC(kernel='linear').fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average="macro"))

conf = confusion_matrix(y_test, y_pred)

plot_conf(model, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred))

df_eval = pd.read_csv('scrapped_pre_processed.csv')

X_eval = count_vec.transform(df_eval["pre_processed"])
df_eval['sentiment_pred_eval'] = model.predict(X_eval)
df_eval.head()
df_eval.to_csv('Scrapped_SentimentAnalysis.csv')