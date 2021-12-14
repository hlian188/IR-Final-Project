import random
from rank_bm25 import BM25Okapi

from myapp.search.objects import ResultItem, Document
def rank_documents_tfidf_cos(terms, docs, index, idf, tf, title_index):

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
    # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    doc_vectors = defaultdict(lambda: [0] * len(terms)) 
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    #HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term]/query_norm * idf[term] 

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26            
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  

    # Calculate the score of each doc 
    # compute the cosine similarity between queyVector and each docVector:
    
    doc_scores=[[np.dot(curDocVec, query_vector) / (np.linalg.norm(curDocVec) * np.linalg.norm(query_vector)), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    result_pred_score = [x[0] for x in doc_scores]

    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_ranking(query, index)
    #print ('\n'.join(result_docs), '\n')
    return result_docs, result_pred_score

def search_ranking(query, index, mode = 'TF-IDF'):
    """
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"                        
            term_docs=[posting[0] for posting in index[term]]
            
            # docs = docs Union term_docs
            docs = docs.union(set(term_docs))
            #docs = set(term_docs)
            #print(docs)
        except:
            #term is not in index
            pass
        

    docs = list(docs)
    if mode == 'TF-IDF':
        ranked_docs, pred_score = rank_documents_tfidf_cos(query, docs, index, idf, tf, title_index)
    else:
        ranked_docs, pred_score = rank_documents_bm25_custom(query, docs, index, tf, title_index)
    return ranked_docs, pred_score


def rank_documents_bm25_custom(terms, docs, index, tf, title_index):
    
    #we use external library to calculate the bm25
    #all algorithms are in this paper http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
    try:
        tokenized_corpus = [title_index[doc_id]['tokens'] for doc_id in docs]
    except:
        print(docs)
    bm25_model = BM25Okapi(tokenized_corpus)
    
    #bm25_score calculates the bm25 score for each documents, given the query vector 'terms'
    bm25_score = bm25_model.get_scores(terms)
    
    #we will now calculate our custom score
    updated_results = []
    for i in range(len(docs)):
        curr_bm25 = bm25_score[i]
        tweet = docs[i]
        
        #initialize the metrics that we will use to calculate custom score
        #explanations and motivations are in the writeup
        
        #curr_length_hashtag_ratio is 1 + log(len(tweet)/(num of hashtags))
        curr_length_hashtag_ratio = 1
        
        #curr_num_likes is 1 + log(len(likes))
        curr_num_likes = 1
        
        #curr_num_retweets is 1 + log(len(retweets))
        curr_retweets = 1
        
        #multiple try clauses in case we divide by 0 or take the logarithm of 0
        try:
            curr_length_hashtag_ratio = 1 + np.log(len(tweet['tokens'])/len(tweet['hashtags']))
        except:
            pass
        
        try:
            curr_num_likes = 1 + np.log(int(tweet['likes']))
        except:
            pass
        
        try:
            curr_retweets = 1 + np.log(int(tweet['retweets']))
            score = curr_bm25 * curr_length_hashtag_ratio * curr_num_likes * curr_retweets
            updated_results.append(score)
        except:
            pass
        return docs, updated_results
    

def build_demo_results(corpus: dict, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)
    ll = list(corpus.values())
    for index in range(random.randint(0, 40)):
        item: Document = ll[random.randint(0, size)]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    # for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
    #     res.append(DocumentInfo(item.Id, item.Tweet, item.Tweet, item.Date,
    #                             "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        results = build_demo_results(corpus, search_id)  # replace with call to search algorithm

        # results = search_in_corpus(search_query)
        ##### your code here #####

        return results
