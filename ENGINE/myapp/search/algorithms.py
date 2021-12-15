import math
import numpy as np
import string
import re
import collections
import nltk

# nltk.download('stopwords')

from collections import defaultdict
from array import array
from numpy import linalg as la
from myapp.search.objects import Document
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# indexCreated = False


def search_in_corpus(query, index_dict):
    """
    Arguments: query is str or list?
    index_dict: dict of the inverted index, tf, df, idf, and title_index
    """
    # 1. create create_tfidf_index
    index, tf, df, idf, title_index = index_dict['index'], index_dict['tf'], index_dict['df'], index_dict['idf'], \
                                      index_dict['title_index']

    # 2. apply ranking
    ranked_docs, pred_score = search_ranking(query, index, tf, idf, title_index)

    return ranked_docs, pred_score, title_index


# tweets are list of documents
def create_tfidf_index(tweets):
    """
    Arguments:
    tweets -- 
    """
    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
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
                current_page_index[term] = [tweet['id'],
                                            array('I', [position])]  # 'I' indicates unsigned int (int in Python)

        # normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document. 
            # posting ==> [current_doc, [list of positions]] 
            # you can use it to infer the frequency of current term.

            # CHECK THIS!
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1]) / norm, 4))  ## SEE formula (1) above
            # increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1  # increment DF for current term

        # merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = 1 + np.round(np.log(float(len(tweets) / df[term])), 4)

    return index, tf, df, idf, title_index


def create_tweets(tweets_json):
    tweet_dict = defaultdict()
    tweets = []

    for key in tweets_json:
        tweet_data = {
            'id': tweets_json[key]['id'],
            'full_text': tweets_json[key]['full_text'],
            'tokens': build_tokens(tweets_json[key]['full_text']),
            'username': tweets_json[key]['user']['name'],
            'date': tweets_json[key]['created_at'],
            'hashtags': [key['text'] for key in tweets_json[key]['entities']['hashtags']],
            'likes': tweets_json[key]['favorite_count'],
            'retweets': tweets_json[key]['retweet_count'],
        }

        # sometimes the tweet url doesn't exist
        try:
            tweet_data['url'] = tweets_json[key]['entities']['media'][0]['url']
        except:
            tweet_data['url'] = None

        try:
            tweet_data['url'] = tweets_json[key]["entities"]["url"]["urls"][0]["url"]  # tweet URL
        except:
            try:
                tweet_data['url'] = tweets_json[key]["retweeted_status"]["extended_tweet"]["entities"]["media"][0][
                    "url"]  # Retweeted
            except:
                tweet_data['url'] = ""

        tweets.append(tweet_data)
    return tweets


def build_tokens(tweet):
    """
    Preprocess the Tweet text by removing stop words, emojis, and punctuation and
    stemming, transforming to lowercase and returning the tokens of the text.

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line -- a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    # stop_words = set(stopwords.words("english"))

    line = tweet

    # transform to lowercase
    line = line.lower()

    # remove non-ASCII terms like emojis and symbols
    line = "".join(c for c in line if c in string.printable)

    # remove punctuation EXCEPT for hashtags (see remove_punct())
    line = remove_punct(line)

    # tokenize the text to get a list of terms
    line = line.split()

    # remove html tags, blank spaces like '', and urls
    line = [word for word in line if
            not (re.match("^qampa$", word) or re.match("^amp$", word) or re.match("^http", word))
            and word]

    # remove standalone numbers e.x. '19' but not the 19 from 'covid19'
    line = [word for word in line if not word.isnumeric()]

    # add standalone word as token too if it has number e.x. 'covid19' gets tokenized as 'covid19' and 'covid'
    line = line + [word.rstrip(string.digits) for word in line if sum([c.isdigit() for c in word]) != 0]

    # remove stopwords
    # line = [word for word in line if word not in stop_words]

    # perform stemming
    line = [stemmer.stem(word) for word in line]

    # add unhashtagged word if it's hashtag is present
    # e.x. if #covid is present, we also add covid as a token
    line = line + [word.replace('#', '') for word in line if word[0] == '#']

    return line


def remove_punct(line):
    """
    Helper function to remove punctuation EXCEPT for '#''

    Arugment:
    line -- string of text

    Returns:
    line -- string of text without punctuation
    """
    return line.translate(str.maketrans('', '', string.punctuation.replace('#', '')))


def search_ranking(query, index, tf, idf, title_index):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_tokens(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs = [posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs = docs.union(set(term_docs))
            # docs = set(term_docs)
            # print(docs)
        except:
            # term is not in index
            pass

    docs = list(docs)

    ranked_docs, pred_score = rank_documents_tfidf(query, docs, index, idf, tf, title_index)

    return ranked_docs, pred_score


def rank_documents_tfidf(terms, docs, index, idf, tf, title_index):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms))  # I call doc_vectors[k] for a nonexistent key k, the key-value
    # pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  # termIndex is the index of the term in the query
        if term not in index:
            continue

        # Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            # tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)

    result_docs = [x[1] for x in doc_scores]
    result_pred_score = [x[0] for x in doc_scores]

    '''# print document titles instead if document id's
    # result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)
    # print ('\n'.join(result_docs), '\n')'''

    return result_docs, result_pred_score
