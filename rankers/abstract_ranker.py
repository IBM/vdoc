from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import List


class AbstractRanker(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def rerank(self, queries: List[str], passages: List[List[dict]], metadata: dict = None):
        pass

    @classmethod
    def argsort(cls, items, key=None, reverse=False):
        new_key = items.__getitem__ if key is None else lambda x: key(items.__getitem__(x))
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(items)), key=new_key, reverse=reverse)

    @classmethod
    def propagate_scores(cls, passages, flat_scores):
        # create a deep copy of the passages and
        # propagate the flat list of scores.
        # sort under each query
        rerank_scores = deque(flat_scores)
        rerank_passages = []
        rerank_orders = []
        for query_passages in passages:
            cls.propagate_query_scores_int(query_passages=query_passages,
                                           scores_queue=rerank_scores,
                                           rerank_passages=rerank_passages,
                                           rerank_orders=rerank_orders)
        return rerank_passages, rerank_orders

    @classmethod
    def propagate_query_scores(cls, query_passages, scores, rerank_passages, rerank_orders):
        rerank_scores = deque(scores)
        cls.propagate_query_scores_int(query_passages=query_passages,
                                       scores_queue=rerank_scores,
                                       rerank_passages=rerank_passages,
                                       rerank_orders=rerank_orders)

    @classmethod
    def propagate_query_scores_int(cls, query_passages, scores_queue: deque, rerank_passages, rerank_orders):
        query_passages = deepcopy(query_passages)
        for passage in query_passages:
            passage['_score'] = scores_queue.popleft()
        sort_order = cls.argsort(query_passages, key=lambda x: x['_score'], reverse=True)
        query_passages[:] = [query_passages[i] for i in sort_order]
        rerank_passages.append(query_passages)
        rerank_orders.append(sort_order)
