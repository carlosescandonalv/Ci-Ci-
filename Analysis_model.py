import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys,os
from collections import Counter
import plotly.express as px
import seaborn as sn

df=pd.read_csv('Scrapped_SentimentAnalysis.csv')

def view_distribution(data, subset):
    plt.figure(figsize=(6,5))
    plt.title("Number of different sentiments - "+subset)
    plot = sn.countplot(x = data['sentiment_pred_eval'], data=data)
    for p in plot.patches:
        plot.annotate(p.get_height(),(p.get_x()+0.1 ,p.get_height()+50))

view_distribution(df,"sentiment_pred_eval")

df_positive=df.loc[df["sentiment_pred_eval"] == 'Positive']
df_negative=df.loc[df["sentiment_pred_eval"] == 'Negative']
#print(df_positive.columns)

#WORDCLOUD
wc = WordCloud(collocations=False,
               background_color='white',
               height = 600,
               width =400,)


palabras= " ".join(word for word in df_positive.pre_processed)
palabras2= " ".join(word for word in df_negative.pre_processed)

wc_pos = wc.generate(palabras)
#wc_neg= wc.generate(palabras2)
plt.show()
plt.figure()
wc_pos.to_file("pos.png")
#wc_neg.to_file("neg.png")
#print(palabras)



#Most common words Positive
df_positive['temp_list'] = df_positive['pre_processed'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_positive['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
print(temp)

fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words - Positive')
fig.show()


#MOST COMMON NEGATIVE
df_negative['temp_list'] = df_negative['pre_processed'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df_negative['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
print(temp)

fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words - Negative')
fig.show()

