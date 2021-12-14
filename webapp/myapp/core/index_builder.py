import json
from collections import defaultdict
from array import array
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import re
import string
import pickle
import sys
def remove_punct(line):
    """
    Helper function to remove punctuation EXCEPT for '#''
    
    Arugment:
    line -- string of text
    
    Returns:
    line -- string of text without punctuation
    """
    return line.translate(str.maketrans('', '', string.punctuation.replace('#', '')))

def build_terms(line):
    """
    Preprocess the Tweet text by removing stop words, emojis, and punctuation and
    stemming, transforming to lowercase and returning the tokens of the text.
    
    Argument:
    line -- string (text) to be preprocessed
    
    Returns:
    line -- a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # transform to lowercase 
    line =  line.lower() 
    
    # remove non-ASCII terms like emojis and symbols
    line = "".join(c for c in line if c in string.printable) 
    
    # remove punctuation EXCEPT for hashtags (see remove_punct())
    line = remove_punct(line)
    
    # tokenize the text to get a list of terms
    line = line.split() 
    
    # remove html tags, blank spaces like '', and urls
    line = [word for word in line if not (re.match("^qampa$" , word) or re.match("^amp$" , word) or re.match("^http" , word)) 
    and word] 
    
    # remove standalone numbers e.x. '19' but not the 19 from 'covid19'
    line = [word for word in line if not word.isnumeric()]
    
    # add standalone word as token too if it has number e.x. 'covid19' gets tokenized as 'covid19' and 'covid'
    line = line + [word.rstrip(string.digits) for word in line if sum([c.isdigit() for c in word]) != 0]
    
    # remove stopwords
    line = [word for word in line if word not in stop_words] 
    
    # perform stemming
    line = [stemmer.stem(word) for word in line]
    
    # add unhashtagged word if it's hashtag is present 
    # e.x. if #covid is present, we also add covid as a token
    line = line + [word.replace('#', '') for word in line if word[0] == '#' ] 
    
    return line

# tweet_dict is our output data structure that maps Tweet IDs to their text
# note we need to keep the following information
# Tweet | Username | Date | Hashtags | Likes | Retweets | Url

def create_tweets(tweets_json):
    """
    Argument: Json of Tweets

    Returns: list of Tweets represented as dictionaries
    """
    
    tweet_dict = defaultdict()
    tweets = []

    for key in tweets_json:
        tweet_data = {
            'id': tweets_json[key]['id'],
            'full_text': tweets_json[key]['full_text'],
            'tokens': build_terms(tweets_json[key]['full_text']),
            'username': tweets_json[key]['user']['name'],
            'date': tweets_json[key]['created_at'],
            'hashtags': [key['text'] for key in tweets_json[key]['entities']['hashtags']],
            'likes': tweets_json[key]['favorite_count'],
            'retweets': tweets_json[key]['retweet_count'], 
        }

        #sometimes the tweet url doesn't exist
        try:
            tweet_data['url'] = tweets_json[key]["entities"]["url"]["urls"][0]["url"]  # tweet URL
        except:
            try:
                tweet_data['url'] = tweets_json[key]["retweeted_status"]["extended_tweet"]["entities"]["media"][0]["url"]  # Retweeted
            except:
                tweet_data['url'] = ""

        """
        try:
            tweet_data['url'] = tweets_json[key]['entities']['media'][0]['url']
        except:
            tweet_data['url'] = None
        """
        
        tweets.append(tweet_data)
    return tweets

# create index
def create_index(tweets_json):
    tweets = create_tweets(tweets_json)
    index = defaultdict(list)
    title_index = defaultdict()

    for tweet in tweets:
        title_index[tweet['id']] = tweet
        
        #current page index keeps track of postision of each word in tweet
        #e.x. if our tweet #50 has tokens "covid health world covid", our current_page_index looks like:
        # {covid -> [50, [0, 3]], health -> [50, [1]], world [50, [2]]}
        current_page_index = {}
        for position, word in enumerate(tweet['tokens']):
            
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list
                current_page_index[word][1].append(position)  
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[word]=[tweet['id'], array('I', [position])] #'I' indicates unsigned int (int in Python)
        

        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)
    
    return index, title_index

# apply tf-idf
# tweets is a list of tokens
def create_tfidf_index(tweets):
    index = defaultdict(list)
    tf = defaultdict(list)  #term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  #document frequencies of terms in the corpus
    title_index = defaultdict()
    idf = defaultdict(float)


    for tweet in tweets:
        
        title_index[tweet['id']] = tweet
        current_page_index = {}

        for position, term in enumerate(tweet['tokens']):  ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[tweet['id'], array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document. 
            # posting ==> [current_doc, [list of positions]] 
            # you can use it to infer the frequency of current term.
            
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm, 4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = 1 + np.round(np.log(float(len(tweets)/df[term])), 4)

    return index, tf, df, idf, title_index


if __name__ == '__main__':

    #write index to pickle file
    tweet_path = 'files/dataset_tweets_WHO.txt'
    with open(tweet_path) as f:
        tweets_json = json.load(f)
    tweets = create_tweets(tweets_json)

    index, tf, df, idf, title_index = create_tfidf_index(tweets)

    result_dict = {'index': index, 'tf': tf, 'idf': idf, 'title_index': title_index}
    
    #create pickle file
    filename = 'files/inverted_index'
    outfile = open(filename,'wb')
    pickle.dump(result_dict ,outfile)
    outfile.close()