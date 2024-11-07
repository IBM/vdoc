import os
import pickle
import hashlib
from typing import List

from rankers.abstract_ranker import AbstractRanker
from rankers.perplexity.perplexity_client import PerplexityRanker
from rankers.tokenizer_utils import TokenizerUtils

unique_ids = set()
class VirtualDocument:

    def __init__(self, reranker: AbstractRanker, tokenizer_utils: TokenizerUtils):
        self.reranker = reranker
        self.tokenizer_utils = tokenizer_utils

    def create(self, prompt: str, document: str,
               query: str, prompt_token_limit: int,
               passage_token_limit: int,
               segments: List[str] = None,
               keep_order = True,
               metadata: dict = None):

        spare = 32

        prompt_length = self.tokenizer_utils.get_length(prompt)

        # no need to segment small document
        if prompt_length > prompt_token_limit:

            # make sure the document is contained inside the prompt
            document_index = prompt.find(document)
            if document_index < 0:
                raise ValueError("Document not in prompt")

            # split into passages (or use given segments) and compute lengths
            if segments:
                passages, (lengths, _) = segments, self.tokenizer_utils.get_lengths(segments)
            else:
                passages, lengths = self.tokenizer_utils.segment(document, passage_token_limit)

            # rankers the passages and obtain the sort order (argsort)
            reranker_passages, reranker_orders = self.reranker.rerank(
                [query], [[{'_text': passage} for passage in passages]], metadata=metadata)

            # there's only one query in our case
            reranker_passages = [passage['_text'] for passage in reranker_passages[0]]
            reranker_orders = reranker_orders[0]
            reranker_lengths = [lengths[i] for i in reranker_orders]

            # the part before the document and after it
            prompt_opening = prompt[0:document_index]
            prompt_ending = prompt[document_index+len(document):]

            # conclude the space available for the document
            document_token_limit = prompt_token_limit - (self.tokenizer_utils.get_length(prompt_opening) +
                                                         self.tokenizer_utils.get_length(prompt_ending)) - spare

            # collect as many passages as possible in descending order of rank,
            # while maintaining the original passage order
            collected_passages = self.tokenizer_utils.collect(texts=reranker_passages,
                                                              max_length=document_token_limit,
                                                              lengths=reranker_lengths,
                                                              max_texts=None)

            # get the ids of the collected passages
            collected_passages_ids = reranker_orders[0:len(collected_passages)]

            if keep_order:
                # get the ids of the collected passages
                collected_passages_ids = sorted(collected_passages_ids)

            # concatenate their content in the original order they appeared
            virtual_document = ' '.join([passages[passage_id].strip()
                                         for passage_id in collected_passages_ids])

            # replace the document in the prompt with the virtual document
            new_prompt = f"{prompt_opening}" \
                         f"{virtual_document}" \
                         f"{prompt_ending}"

            return collected_passages_ids, virtual_document, new_prompt

        # no need to segment, so return the document/segments as is
        if segments:
            document = ' '.join(segments)

        return None, document, prompt


def create_vdoc_prompt(document, query):
    instruction = "Please complete the conversation by adding an agent turn"
    return f"{instruction}\n\n{document}\n\n{query}\nagent:"


def get_vdoc_example():

    with open(os.path.join("resources", "document.txt"), "rt") as f:
        document = f.read()

    with open(os.path.join("resources", "query.txt"), "rt") as f:
        query = f.read()

    return {
        "document": document,
        "query": query,
        "prompt_creator": create_vdoc_prompt
    }


def main():
    vdoc_example = get_vdoc_example()
    document = vdoc_example['document']
    query = vdoc_example['query']
    prompt = vdoc_example['prompt_creator'](document, query)

    language_model = 'google/flan-t5-small'

    ranker = PerplexityRanker(model_name=language_model)
    tokenizer_utils = TokenizerUtils(model=language_model)
    vdoc_creator = VirtualDocument(ranker, tokenizer_utils)
    prompt_token_limit = 1500
    _, vdoc, new_prompt = vdoc_creator.create(
        prompt=prompt, document=document,
        query=query, passage_token_limit=128,
        prompt_token_limit=prompt_token_limit)

    document_length = tokenizer_utils.get_length(document)
    vdoc_length = tokenizer_utils.get_length(vdoc)

    print(f"input document length: {document_length}")
    print(f"v-doc length for window size of {prompt_token_limit} is {vdoc_length}")

    vdoc_prompt = vdoc_example['prompt_creator'](vdoc, query)
    assert vdoc_prompt == new_prompt

if __name__ == "__main__":
    main()
