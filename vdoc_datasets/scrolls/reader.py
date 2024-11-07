import jsonlines
from vdoc_datasets.record import record
from utils import create_unique_pid, create_unique_url
from rankers.tokenizer_utils import TokenizerUtils

PASSAGE_LEN = 512 #todo: pass this from outside

class scrolls_reader():

    def __init__(self, input_file, document_mode=False):
        self.input_file = input_file
        self.document_mode = document_mode
        self.dataset_name = 'scrolls'
        self.tokenizer_utils = TokenizerUtils()

    def read(self):
        input_file = self.input_file
        with jsonlines.open(input_file, 'r') as f:
            for line in f:
                doc_id = line['id']
                _, text = parse_input(line)

                if self.document_mode:
                    yield record(doc_id, url=doc_id,
                            title='', text=text, collection=self.dataset_name)
                else:
                    #change to_evidence_x to get all passages (for indexing)
                    if 'evidence' in line:
                        # take only the first evidence
                        passages = line['evidence'][:1]
                    else:
                        passages, _ = self.tokenizer_utils.segment(text, passage_token_limit=PASSAGE_LEN)

                    passage_id = 0
                    for passage in passages:
                        pid = create_unique_pid(doc_id, passage_id)
                        yield record(pid, url=create_unique_url(doc_id, passage_id),
                                        title='', text=passage, collection=self.dataset_name,
                                        productName=doc_id)
                        passage_id += 1

    # default filtering - keep all records
    def filter_records(self, records):
        return records


def parse_input(line):
    input = line['input']
    if 'query_start_index' in line and 'document_start_index' in line:
        document = input[line['document_start_index']:line['document_end_index']]
        query = input[line['query_start_index']:line['query_end_index']]
        #query = remove_query_header(query, 'Question:\n')
        #query = remove_query_header(query, 'Query:\n')
    else:
        sep = input.find('\n\n')
        query = input[0:sep]

        document = input[sep+2:]
    return query, document

def remove_query_header(query, header='Question:\n'):
    query_start = query.find(header)
    if query_start >= 0:
        query = query[len(header):]
    return query


