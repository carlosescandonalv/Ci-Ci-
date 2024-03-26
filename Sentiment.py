import pandas as pd
import numpy as np
from cleantext import clean
from sklearn.model_selection import train_test_split
import re

df=pd.read_csv('pre_processed.csv')
print(df.shape)

df = clean(df,no_emoji=True)  #remove emojis
#df = re.sub(r'pic.twitter.com/[\w]*',"", df)
df.drop_duplicates(['pre_processed'], inplace = True)


df.to_csv('pre_processedv2.csv', index=False)

print(df.shape)
print(df.info)


X=df['index']
y=df['pre_processed']
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3)