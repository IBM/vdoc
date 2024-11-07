import os
import pickle
import json
from gzip import GzipFile
from utils import remove_non_alphabetic, create_unique_pid, create_unique_url

class googlenq_benchmark():

    def __init__(self, input_file, document_mode=False):
        self.input_file = input_file
        self.document_mode = document_mode
        self.dataset_name = 'googlenq'

    # read the queries from the annotated files. For each query, keep only a single gold long
    # answer (i.e., a single passage) keep it in cache
    # we run on the train files because we use googlenq to fine-tune models

    def get_queries(self):
        queries,_,_ = self.get_benchmark()
        return queries

    # read the qrels from the annotated files. For each document, keep only a gold
    # single long answer (i.e., a single passage) keep it in cache
    # we run on the train files because we use googlenq to fine-tune models

    def get_qrels(self):
        _,qrels,_ = self.get_benchmark()
        return qrels

    # for each record, get the short answer
    def get_target_responses(self):
        _, _, responses = self.get_benchmark()
        return responses


    def get_benchmark(self):
        cache_file = os.path.join('resources', self.dataset_name, 'benchmark_cache') if self.document_mode else os.path.join('resources', self.dataset_name, 'passages_benchmark_cache')
        try:
            benchmark = pickle.load(open(cache_file, mode="rb"))
            print("benchmark loaded from cache")
        except:
            benchmark = self._get_benchmark()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pickle.dump(benchmark, open(cache_file, mode="wb"))
        return benchmark

    def _get_benchmark(self):

        queries = dict()
        qrels = dict()
        responses = dict()
        grounding = dict()

        with open(self.input_file, 'rb') as f:
            count = 0
            with GzipFile(fileobj=f) as inf:
                for line in inf:
                    # each line contains a query
                    count += 1
                    if count % 100 == 0:
                        print ('read_records ' + str(count))
                    obj = json.loads(line.decode("utf-8"))
                    # if no exception is thrown, obj contains a record for a technote
                    try:
                        id = str(obj['example_id'])
                        doc_id = normalize_title(obj['document_title'])
                        annotations = obj['annotations']
                        tokens = obj['document_tokens']
                        query = obj['question_text']
                        passage_id = 0
                        for annotation in annotations:
                            short_answers = annotation['short_answers']
                            long_answer = annotation['long_answer']
                            long_start_token = long_answer['start_token']
                            long_end_token = long_answer['end_token']

                            # skip yes_no questions
                            # check also that end_token is larger than start_token
                            if annotation["yes_no_answer"] == 'NONE' and 0 <= long_start_token < long_end_token:
                                passage_tokens = tokens[long_start_token:long_end_token]
                                passage_type = passage_tokens[0]['token'] if passage_tokens[0]['html_token'] else ''
                                # consider only long answers of type posaages and queries that have a short answer
                                if passage_type == '<P>' and short_answers:

                                    # match the gold passage with a semantic segment
                                    pid = create_unique_url(doc_id, passage_id)

                                    # queries
                                    queries[id] = query

                                    # qrels
                                    qrel = qrels[id] if id in qrels else list()
                                    qrels[id] = qrel

                                    # for passage mode, we can have multiple gold passages
                                    qrel.append(doc_id if self.document_mode else pid)
                                    passage_id += 1

                                    # responses
                                    for short_answer in short_answers:
                                        short_start_token = short_answer['start_token']
                                        short_end_token = short_answer['end_token']
                                        response = responses[id] if id in responses else list()
                                        responses[id] = response
                                        response.append(' '.join([token['token'] for token in tokens[short_start_token:short_end_token]]))


                    except Exception as e:
                        print (e)
        return queries, qrels, responses

def normalize_title(title):
    return remove_non_alphabetic(title).lower()

