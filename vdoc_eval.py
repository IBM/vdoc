import argparse
import csv
import os

from create_virtual_document import VirtualDocument
from generation.generation_utils import is_perplexity
from rankers.elser.elser_client import ElserRanker
from rankers.perplexity.perplexity_client import PerplexityRanker
from rankers.prefix.prefix_client import PrefixRanker
from rankers.sbert.sbert_client import SBertRanker
from rankers.tokenizer_utils import TokenizerUtils
from utils import included_overlap
from vdoc_datasets.googlenq.benchmark import googlenq_benchmark
from vdoc_datasets.googlenq.reader import googlenq_reader
from vdoc_datasets.scrolls.benchmark import scrolls_benchmark
from vdoc_datasets.scrolls.reader import scrolls_reader

import logging
logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log format
    )
logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# direct vdoc evaluation. Check if gold passage is in vdoc
def direct_vdoc_evaluation(records, queries, qrels, responses,
                           gold_passages=None, out_filename=None,
                           queries_limit=-1,
                           language_model=None,
                           max_new_tokens=0, model_token_limit=2400, passage_token_limit=512,
                           order='doc'
                           ):

    tokenizer_utils = TokenizerUtils()
    reranker = get_ranker(language_model)
    vdoc_creator = VirtualDocument(reranker, tokenizer_utils)
    model_token_limit = model_token_limit
    generation_token_limit = max_new_tokens
    prompt_token_limit = model_token_limit - generation_token_limit

    with open(out_filename, "w", encoding="utf-8") as fout:

        csvwriter = csv.writer(fout, delimiter=',')
        # write headers
        csvwriter.writerow(['doc_id', 'qid', 'title', 'gold passage', 'content', 'query', 'response'])
        count = 0
        count_vdoc = 0
        count_bad_vdoc = 0
        count_bad_document = 0

        for query_id, query in queries.items():
            doc_id = list(qrels[query_id])[0]
            document1 = records[doc_id]

            response = list(responses[query_id])[0] if responses else None

            # check for existing segmentation
            segments = document1.passages
            if segments:
                document1 = '\n'.join(segments)

                # for prefix, ignore the semantic segments
                if language_model == 'prefix':
                    segments = None

            count += 1
            if count < 1000:
                if count % 100 == 0:
                    logger.info('Processing ' + str(count))
            else:
                if count % 1000 == 0:
                    logger.info('Processing ' + str(count))

            if queries_limit > 0 and count >= queries_limit:
                break

            # create a virtual document
            document1 = document1.text
            ids, vdoc1, prompt = vdoc_creator.create(prompt=document1,
                                                    query=query,
                                                    document=document1,
                                                    segments=segments,
                                                    prompt_token_limit=prompt_token_limit,
                                                    passage_token_limit=passage_token_limit,
                                                    keep_order=order=='doc',
                                                    metadata={'doc_id': [doc_id]})

            if len(document1) > len(vdoc1):
                    count_vdoc += 1

            gold_passage = gold_passages[query_id][0].text if gold_passages and gold_passages[query_id] else None

            # check if vdoc contains the gold passage (if exist)
            if gold_passage:
                # sanity check that the full document contains the gold passage
                if included_overlap(gold_passage, document1, full_inclusion=True) < 1.0:
                    logger.warning(f'query {query_id} does not have gold passage in original document. Skipping')
                    count_bad_document += 1
                    continue

                # check if vdoc was activated
                if len(document1) > len(vdoc1):
                    passage_vdoc_overlap = included_overlap(gold_passage, vdoc1, full_inclusion=False)

                    # we allow overlap of 90% as a success
                    if passage_vdoc_overlap <= 0.9:
                        count_bad_vdoc += 1

            csvwriter.writerow([doc_id, query_id, '', gold_passage if gold_passage else '', vdoc1, query, response])

        print(f'{out_filename} - Total count {count}. bad documents {count_bad_document}. total vdoc {count_vdoc}. '
              f'bad vdocs {count_bad_vdoc}')


# get the grounding passages for each query
def get_gold_passages(input_file, dataset='scrolls'):
    gold_passages = dict()

    match dataset:
        case 'scrolls':
            reader = scrolls_reader(input_file=input_file, document_mode=False)
            benchmark = scrolls_benchmark(input_file=input_file, document_mode=False)
        case 'googlenq':
            reader = googlenq_reader(input_file=input_file, document_mode=False)
            benchmark = googlenq_benchmark(input_file=input_file, document_mode=False)

    qrels = benchmark.get_qrels()

    # initialize the gold passages
    for qid, qrel in qrels.items():
        gold_passages[qid] = [None for q in qrel]

    # check if we have at least one qrel
    if list(qrels.values())[0]:
        records_dict = dict()
        records = list(reader.read())

        if records:
            gold_passages = dict()
            for record in records:
                # print (record.url + ' - ' + record.title)
                records_dict[record.url] = record

            for qid, qrel in qrels.items():
                gold_passages[qid] = [records_dict[q] if q in records_dict else None for q in qrel]

    return gold_passages


def print_records_statistics(records, tokenizer_utils):
    lens = list()
    for record in records.values():
        if record:
            if isinstance(record, list):
                record = record[0]
        if record:
            lens.append(tokenizer_utils.get_length(record.text))
    lens,_ = tokenizer_utils.get_lengths([record[0].text for record in records.values()])
    if lens:
        print( f'Total counts {len(records)} Min len:  {min(lens)}  Max len:  {max(lens)}  Avg len: {sum(lens) / len(lens)}')


# get reranker (perplexity or sbert)
def get_ranker(model_or_model_name_or_path):

    # simple prefix vdoc
    if model_or_model_name_or_path == 'prefix':
        return PrefixRanker()

    elif model_or_model_name_or_path == 'elser':
        return ElserRanker(dict())

    # perplexity or sbert
    return PerplexityRanker(model_or_model_name_or_path) \
        if is_perplexity(model_or_model_name_or_path) \
        else SBertRanker(model_or_model_name_or_path)


def vdoc(dataset,
         input_file,
         output_file,
         model_name,
         model_token_limit,
         max_new_tokens,
         passage_len,
         order,
         queries_limit):

    # create dir of output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    match dataset:
        case 'scrolls':
            reader = scrolls_reader(input_file=input_file,
                                    document_mode=True)
            benchmark = scrolls_benchmark(input_file=input_file, 
                                          document_mode=True)
        case 'googlenq':
            reader = googlenq_reader(input_file=input_file, 
                                     document_mode=True)
            benchmark = googlenq_benchmark(input_file=input_file, 
                                           document_mode=True)

    records = list(reader.read())
    records_dict = dict()
    for record in records:
        records_dict[record.url] = record

    queries = benchmark.get_queries()
    qrels = benchmark.get_qrels()
    target_responses = benchmark.get_target_responses()
    gold_passages = get_gold_passages(input_file=input_file, dataset=dataset)

    direct_vdoc_evaluation(records_dict, queries, qrels, 
                           target_responses,
                           gold_passages=gold_passages, 
                           out_filename=output_file,
                           language_model=model_name,
                           max_new_tokens=max_new_tokens,
                           model_token_limit=model_token_limit,
                           passage_token_limit=passage_len,
                           order=order,
                           queries_limit=queries_limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name (scrolls/googlenq)",
                        type=str, default="scrolls")
    parser.add_argument("--input_file", help="input file",
                        type=str, required=True)
    parser.add_argument("--output_file", help="output file",
                        type=str, required=True)
    parser.add_argument('--model_name', help='name of model for vdoc',
                        type=str, required=True)
    parser.add_argument('--model_token_limit', help='model max window size',
                        default=4096, type=int)
    parser.add_argument('--max_new_tokens', help='max new generated tokens',
                        default=256, type=int)
    parser.add_argument('--passage_len', help='passage len',
                        default=512, type=int)
    parser.add_argument("--order", help="order of passages in vdoc (doc/rank). Default is doc",
                        type=str)
    parser.add_argument("--queries_limit", help="Number of queries to process. Default is -1 (all queries)",
                        default=-1, type=int)
    args = parser.parse_args()

    vdoc(dataset=args.dataset,
         input_file=args.input_file,
         output_file=args.output_file,
         model_name=args.model_name,
         model_token_limit=args.model_token_limit,
         max_new_tokens=args.max_new_tokens,
         passage_len=args.passage_len,
         order=args.order,
         queries_limit=args.queries_limit)


if __name__ == "__main__":
    main()
