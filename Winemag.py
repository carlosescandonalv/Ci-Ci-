import pandas as pd
from statistics import mean
cols=['country','description','designation','points','price','province','region_1','region_2','variety','winery']
df_1 = pd.read_csv('winemag-data_first150k.csv',usecols=cols)
df_2 = pd.read_csv('winemag-data-130k-v2.csv',usecols=cols)
print(df_1.shape)
#print(df_1.columns)
print(df_2.shape)
#print(df_2.columns)

df = pd.concat([df_1,df_2])
print(df['country'].nunique())
print(df.columns)

#top 10 countries
top10 = df['country'].value_counts()[0:7]
top10v=df['country'].value_counts().head(7).index
print(top10)

#top 10 varieties
top10_variety = df['variety'].value_counts()[0:7]
top10_vari=df['variety'].value_counts().head(7).index
print(top10_variety)
print(top10_vari)

df_top10 = df[df['country'].isin(top10v)]
print(top10v)
print(df_top10.info)

def average_rating(data):
    for c in top10v:
        vec=[]
        df_c = data.loc[data['country'] == c]
        c_mean = df_c["points"].mean()
        print(c, "average rate is:", c_mean)
        #vec.append(c)
        #vec.append(c_mean)
        #print(vec)

print(average_rating(df_top10))

def varieties_per_country(data):

    for c in top10v:
        vec = []
        df_c = data.loc[data['country'] == c]
        varieties = df_c['variety'].value_counts().head(3).index
        #print(c,varieties)
        for i in varieties:
            df_c_v = df_c.loc[df_c['variety'] == i]
            c_mean = df_c_v["points"].mean()
            item= i + ':' + str(c_mean)
            vec.append(item)
            #print(c,":",i,c_mean)
        print(c, ":", vec)

print(varieties_per_country(df_top10))

