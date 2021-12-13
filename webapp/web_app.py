import nltk
from flask import Flask, render_template
from flask import request

from app.analytics.analytics_data import AnalyticsData, Click
from app.core import utils
from app.search_engine.search_engine import SearchEngine
#from ipynb.fs.full.IRWA_Part3_Ranking import build_terms

app = Flask(__name__)

searchEngine = SearchEngine()
analytics_data = AnalyticsData()
corpus = utils.load_documents_corpus()


@app.route('/')
def search_form():
    print(request.headers.get('User-Agent'))
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    results = searchEngine.search(search_query)
    found_count = len(results)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')
    clicked_doc_id = int(request.args["id"])
    analytics_data.fact_clicks.append(Click(clicked_doc_id, "some desc"))

    print("click in id={} - fact_clicks len: {}".format(clicked_doc_id, len(analytics_data.fact_clicks)))

    return render_template('doc_details.html')


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    ### Start replace with your code ###
    docs = []
    for clk in analytics_data.fact_clicks:
        docs.append((corpus[clk.doc_id]))

    return render_template('stats.html', clicks_data=docs)
    ### End replace with your code ###

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port="8088", host="0.0.0.0", threaded=False, debug=True)
