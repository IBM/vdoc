from unitxt.metrics import Perplexity
from rankers.abstract_ranker import AbstractRanker

class PerplexityRanker(AbstractRanker):
    def __init__(
        self,
        model_name: str = 'google/flan-t5-small',
        prompt: str = "Generate a conversation between a user and an agent based on the given content:",
    ) -> None:
        self.model_name = model_name
        self._metric = Perplexity(model_name=model_name,
                                  source_template=f"{prompt} {{reference}}",
                                  target_template="{prediction}")

    def get_name(self):
        return f'perplexity-{self.model_name}'

    def rerank(self, queries: list[str], passages: list[list[str]], metadata:dict = None):
        # Step 0: Flatten queries and passsages as per `unitxt` requirements
        flat_queries = []
        flat_passages = []
        for query, query_passages in zip(queries, passages):
            for passage in query_passages:
                flat_queries.append(query)
                flat_passages.append([passage["_text"]])

        # Step 1: compute P(Q|P) and store in queue
        flat_scores = [
            entry["perplexity"]
            for entry in self._metric.compute(
                references=flat_passages,
                predictions=flat_queries,
                task_data=None
            )
        ]

        # Step 2: create a deep copy of the passages and propagate the scores sort under each query
        return self.propagate_scores(passages, flat_scores)