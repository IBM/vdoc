import os
import jsonlines

from utils import create_unique_url
from vdoc_datasets.scrolls.reader import parse_input

class scrolls_benchmark():
    # currently we have only documents
    def __init__(self, input_file, document_mode=False):
        self.input_file = input_file
        self.document_mode = document_mode
        self.dataset_name = 'scrolls'

    def get_queries(self):
        queries,_,_ = self.get_data()
        return queries

    def get_qrels(self):
        _,qrels,_ = self.get_data()
        return qrels

    def get_target_responses(self):
        _,_,responses = self.get_data()
        return responses

    def get_data(self):
        queries = dict()
        qrels = dict()
        responses = dict()

        input_file = self.input_file
        with jsonlines.open(input_file, 'r') as f:
            for line in f:
                id = line['id']
                pid = line['pid']
                query, _ = parse_input(line)
                queries[pid] = query

                # set qrel to gold passages
                qrel = set()
                if self.document_mode:
                    qrel.add(id)
                else:
                    if 'evidence' in line:
                        passages = line['evidence']
                        passage_id = 0
                        # take only the first evidence
                        for passage in passages[:1]:
                            unique_url = create_unique_url(id, passage_id)
                            qrel.add(unique_url)
                            passage_id += 1

                qrels[pid] = qrel
                if line['output']:
                    responses[pid] = [line['output']]

        return queries, qrels, responses


