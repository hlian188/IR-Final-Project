class AnalyticsData:
    fact_clicks = []
    fact_two = []
    fact_three = []


class Click:
    def __init__(self, doc_id, description):
        self.doc_id = doc_id
        self.description = description
