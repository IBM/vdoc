import json
from typing import List

from sentence_transformers import SentenceTransformer, util as sbert_util

from rankers import ranker_example
from rankers.abstract_ranker import AbstractRanker


class SBertRanker(AbstractRanker):

    def __init__(self, model, batch_size=64):
        self.batch_size = batch_size
        self.sbert_model = SentenceTransformer(model)
        self.model_name = model

    def get_name(self):
        return f'sbert-{self.model_name}'

    def rerank(self, queries: List[str], passages: List[List[dict]], metadata:dict = None):

        # obtain s-bert embeddings for the queries
        queries_emb = self.sbert_model.encode(queries, convert_to_tensor=True)

        rerank_passages = []
        rerank_orders = []

        # loop over each query and its passages
        for query_emb, query_passages in zip(queries_emb, passages):

            # extract the passages text
            query_passages_text = [p['_text'] for p in query_passages]

            # obtain s-bert embeddings for the query passages
            query_passages_emb = self.sbert_model.encode(
                query_passages_text, convert_to_tensor=True)

            # compute cos-sim relative to the query
            result = sbert_util.cos_sim(query_emb, query_passages_emb)

            # propagate the scores and reorder the passages
            self.propagate_query_scores(query_passages=query_passages, scores=result[0].tolist(),
                                        rerank_passages=rerank_passages, rerank_orders=rerank_orders)

        return rerank_passages, rerank_orders


def main():
    reranker = SBertRanker(model='sentence-transformers/all-mpnet-base-v2',
                           batch_size=10)

    reranker_results, _ = \
        reranker.rerank(queries=reranker_example.queries,
                        passages=reranker_example.passages)


    print(json.dumps(reranker_results, indent=4))


if __name__ == '__main__':
    main()
