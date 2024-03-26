import pandas as pd
import numpy as np
from cleantext import clean
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import seaborn as sn
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

df=pd.read_csv('twitter_training.csv')
print(df['text'])
from cleantext import clean

# preprocessing text
def preprocessing(string):
    #result = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(string))  # links
    result = re.sub('http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(string))  # links
    result = re.sub('pic?.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+',"",result) #remove pic.twitter
    result = re.sub(r'#', '', result)  # remove hashtags
    result = re.sub(re.compile('<.*?>'),'', result)  # remove HTML tags
    result = re.sub(r'@[A-Za-z0-9]+',' ', result)  # remove mentions
    result = re.sub('[^A-Za-z0-9]+', ' ', result)  # remove special characters
    result = clean(result,no_emoji=True)  #remove emojis
    return result

# duplicates removal
df.drop_duplicates(['text'], inplace = True)

df['text']=df['text'].apply(lambda cw : preprocessing(cw))


# stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#lematizing
lemmatizer = WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word,'v') for word in x.split()]))

# Processed test
df.to_csv('pre_processedv2.csv', index=False)

print(df.shape)
print(df.info)


#X=df['index']
#y=df['pre_processed']
#X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3)