import snscrape.modules.twitter as sntwitter
import pandas as pd

#query = "(wine OR redwine OR whitewine) (#wine OR #redwine OR #whitewine) lang:en until:2021-12-31 since:2021-01-01 -filter:links -filter:replies"
#query = "(LaRiojaAlta OR Tignanello OR Beringer OR FamiliaTorres OR Catena OR Henschke OR Penfolds OR CVNE OR Antinori OR Château OR ChâteauMusar OR Symington OR Gaja OR Esporão OR Sassicaia) (#LaRiojaAlta OR #FamiliaTorres OR #riberaduero OR #Tignanello OR #Beringer OR #VegaSicilia OR #Henscke OR #Penfolds OR #RamonBilbao) lang:en"
#query = "wine (Spain OR Spanish) (#Spanishwine OR #Spanishwines) lang:en"
#query = "wine (Portugal OR Portuguese) (#Portuguesewine OR #Portuguesewines) lang:en"
#query = "wine Port (#Port) lang:en"
query = "(wine OR redwine OR whitewine) (#wine #redwine #whitewine) lang:en"
tweets = []
limit = 50000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['User', 'Tweet'])
print(df)

# to save to csv
df.to_csv('tweets_scrapped.csv')
