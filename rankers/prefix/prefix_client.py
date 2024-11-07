import json
from typing import List
from rankers import ranker_example
from rankers.abstract_ranker import AbstractRanker



class PrefixRanker(AbstractRanker):

    DEF_BATCH_SIZE = 32

    def __init__(self, stride=10, progress_bar=False):
        self.stride = stride
        self.progress_bar = progress_bar

    def get_name(self):
        return 'prefix'

    def rerank(self, queries: List[str], passages: List[List[dict]], metadata: dict= None):
        # return the passages by their order
        flat_scores = []
        for query, query_passages in zip(queries, passages):
            flat_scores += [-i for i in range(len(query_passages))]
        return self.propagate_scores(passages, flat_scores)


def main():
    reranker = PrefixRanker()

    reranker_results, reranker_orders = \
        reranker.rerank(queries=reranker_example.queries,
                        passages=reranker_example.passages)

    print(json.dumps(reranker_results, indent=4))


if __name__ == '__main__':
    main()
