import pickle
import json

from myapp.search.algorithms import create_tfidf_index
from myapp.search.algorithms import create_tweets

#when file is run, we generate inverted index as pickle file
#i.e. we are storing the index in memory so we do not recompute it for every search query

path = 'tweets-data-who.json'
with open(path) as f:
    tweets_json = json.load(f)
tweets = create_tweets(tweets_json)

(index, tf, df, idf, title_index) = create_tfidf_index(tweets)
results_dict = {'index': index, 'tf': tf, 'df': df, 'idf': idf, 'title_index': title_index}
outfile = open('inverted_index', 'wb')
pickle.dump(results_dict, outfile)
outfile.close()
