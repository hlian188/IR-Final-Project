import datetime
from random import random

from faker import Faker

fake = Faker()


# fake.date_between(start_date='today', end_date='+30d')
# fake.date_time_between(start_date='-30d', end_date='now')
#
# # Or if you need a more specific date boundaries, provide the start
# # and end dates explicitly.
# start_date = datetime.date(year=2015, month=1, day=1)
# fake.date_between(start_date=start_date, end_date='+30y')

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )


class Document:
    def __init__(self, id, title, description, doc_date, email, ip):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.email = email
        self.ip = ip


def load_documents_corpus():
    """
    Load documents corpus from dataset_tweets_WHO.txt file
    :return:
    """

    ##### demo replace ith your code here #####
    docs = []
    for i in range(200):
        docs.append(Document(fake.uuid4(), fake.text(), fake.text(), fake.date_this_year(), fake.email(), fake.ipv4()))
    return docs
