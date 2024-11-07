import os
from rankers.abstract_ranker import AbstractRanker
from elasticsearch import Elasticsearch
from ssl import create_default_context
from collections import defaultdict

class ElserRanker(AbstractRanker):
    def __init__(
        self,
        settings: dict) -> dict():

        es_api_key = os.getenv("ES_API_KEY")
        es_ssl_fingerprint = os.getenv("ES_SSL_FINGERPRINT")
        host = os.getenv("ES_HOST")
        auth = os.getenv("ES_AUTH")
        index_name = os.getenv("ES_INDEX")

        self.index_name = index_name
        auths = auth.strip().split(':') if auth else None
        self.auth = (auths[0].strip(), auths[1].strip()) if auths and len(auths) == 2 else None

        self.text_field = settings['text_field'] if 'text_field' in settings else 'text'
        self.title_field = settings['title_field'] if 'title_field' in settings else 'title'
        self.filter_field = settings['filter_field'] if 'filter_field' in settings else 'productId'

        if es_ssl_fingerprint and es_api_key:
            self.client = Elasticsearch(
                hosts = [host],
                ssl_assert_fingerprint=es_ssl_fingerprint,
                api_key=es_api_key,
                request_timeout=60
            )
        elif es_ssl_fingerprint:
            context = create_default_context(cafile=es_ssl_fingerprint)
            self.client = Elasticsearch(
                hosts = [host],
                basic_auth=(self.auth[0], self.auth[1]),
                ssl_context=context,
                request_timeout=480
            )
        else:
            self.client = Elasticsearch(
                hosts=[host],
                basic_auth=(self.auth[0], self.auth[1])
            )

        self.model_name = ".elser_model_1"

        self.processors = [
            {
                "inference": {
                    "model_id": self.model_name,
                    "target_field": "ml",
                    "field_map": {
                        "text": "text_field"
                    },
                    "inference_config": {
                        "text_expansion": {
                            "results_field": "tokens"
                        }
                    }
                }}
        ]

    def get_name(self):
        return 'elser'

    def create_query(self, query):
        return {
            "bool": {
                "must": {
                    "text_expansion": {
                        "ml.tokens": {
                            "model_id": self.model_name,
                            "model_text": query
                        }
                    }
                },
            }
        }

    def multi_field_search(self, query,
                           doc_id=None,
                           size=40,
                           remove_dups=True):
        es_query = self.create_query(query)

        if self.filter_field and doc_id:
            es_query['bool']["filter"] = {
                    "term": {self.filter_field: doc_id}
            }

        res = self.client.search(
            index=self.index_name,
            query=es_query,
            size=size,
        )
        status_code = res.meta.status

        # we want to retain the original order of passages in the document
        # one option is to match the retrieved passages to the original passages using text comparison
        # this means that we must have full sync between the given passages (e.g., from a json file)
        # to the passages in the index.
        # alternatively, we can assume that passage_ids have a suffix (_<num>) that we can use
        # to obtain original order
        passages = self.to_passages(res._body['hits']['hits'],
                                    text_field=self.text_field,
                                    title_field=self.title_field,
                                    remove_dups=remove_dups,
                                    doc_id=doc_id)

        return passages


    # pass the doc_id filter in the passages parameter
    def rerank(self, queries: list[str], passages: list[list[str]], metadata: dict = None):
        # we do a retrieval and ignore the input passages
        flat_scores = []
        retrieved_passages = []
        doc_ids =  metadata['doc_id'] if metadata and 'doc_id' in metadata else [None] * len(queries)
        for query, _passages, doc_id in zip(queries, passages, doc_ids):
            query_passages = self.multi_field_search(query, doc_id=doc_id, size=len(_passages))
            # sort by passage_id
            if doc_id:
                query_passages = sorted(query_passages, key=lambda k: k['passage_id'], reverse=False)

            flat_scores += [passage['_score'] for passage in query_passages]
            retrieved_passages.append(query_passages)

        return self.propagate_scores(retrieved_passages, flat_scores)

    def to_passages(self, hits, text_field='text', title_field='title', remove_dups=True, doc_id=None):
        passages = {}
        for hit in hits:
            passage_id = f"{hit['_id']}"

            # if we got a doc_id, sort the passages by their passage_id (assuming a suffix _<num>)
            if doc_id:
                passage_id = int(passage_id[passage_id.rfind("-")+1:])
            text = hit['_source'][text_field]
            if isinstance(text, list):
                text = ' '.join([v for v in text])
            passages[passage_id] = {
                                    'passage_id': passage_id,
                                    '_text': text,
                                    '_score': hit['_score']}
        if remove_dups:
            texts = defaultdict(list)
            for passage_id, passage_data in passages.items():
                text = passage_data['_text']
                texts[text].append(passage_id)
            passages = [passages[sorted(passage_ids)[0]]
                        for text, passage_ids in texts.items()]
        return passages
