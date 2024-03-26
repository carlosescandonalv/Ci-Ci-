import pandas as pd

df=pd.read_csv('evaluation.csv')
print(df.columns)

print(df['description'])

count=df['sentiment_pred_eval'].value_counts()
print(count)


df_positive= df.loc[df['sentiment_pred_eval'] == 'Positive']
topcountries= df_positive['country'].value_counts()

country_top3= df_positive['country'].value_counts().head(5).index
print(topcountries)
print(country_top3)

topvarieties= df_positive['variety'].value_counts()
variety_top3= df_positive['variety'].value_counts().head(5).index
print(topvarieties)
print(variety_top3)

def average_rating(data):
    for c in country_top3:
        df_c = data.loc[data['country'] == c]
        c_mean = df_c["points"].mean()
        price = df_c["price"].mean()
        print(c, "average rate is:", c_mean, "Price:",price, "$")

print(average_rating(df_positive))

def varieties_per_country(data):
    for c in country_top3:
        vec = []
        df_c = data.loc[data['country'] == c]
        varieties = df_c['variety'].value_counts().head(3).index
        #print(c,varieties)
        for i in varieties:
            df_c_v = df_c.loc[df_c['variety'] == i]
            c_mean = df_c_v["points"].mean()
            price_mean = df_c_v['price'].mean()
            item= i + ':' + str(c_mean) + ' || price:' +str(price_mean) + '$'
            vec.append(item)
            #print(c,":",i,c_mean)
        print(c, ":", vec)

print(varieties_per_country(df_positive))
#print(df_positive['designation'].value_counts())