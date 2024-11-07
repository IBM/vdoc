import os
import json
import pickle
from gzip import GzipFile
from vdoc_datasets.record import record
from vdoc_datasets.googlenq.benchmark import normalize_title
from utils import create_unique_pid, create_unique_url
from vdoc_datasets.googlenq.benchmark import googlenq_benchmark
from bs4 import BeautifulSoup
from rankers.tokenizer_utils import TokenizerUtils

PASSAGE_LEN=512 #todo: pass this from outside
VDOC_MODE=True

class googlenq_reader():

    def __init__(self, input_file, document_mode=False):
        self.input_file = input_file
        self.document_mode = document_mode
        self.dataset_name = 'googlenq'
        self.tokenizer_utils = TokenizerUtils()

    def read(self):
        cache_file = os.path.join('resources',self.dataset_name, 'docs_cache') if self.document_mode else os.path.join('resources',self.dataset_name,'passages_cache')
        try:
            data = pickle.load(open(cache_file, mode="rb"))
            print("data loaded from cache")
        except:
            data = list(self._read())
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pickle.dump(data, open(cache_file, mode="wb"))
        return data

    # we limit ourselves to question where the gold answer is a passage
    # for that reason, we read first the qrels and then read only questions that appear in the qrels
    def _read(self):
        benchmark = googlenq_benchmark(self.input_file, document_mode=self.document_mode)

#       input_paths = glob.glob(self.dataset_dir)
        qrels = benchmark.get_qrels()

        with open(self.input_file, 'rb') as f:
            count = 0
            with GzipFile(fileobj=f) as inf:
                for line in inf:
                    # each line contains a query
                    count += 1
                    if count % 100 == 0:
                        print ('read_paragraphs ' + str(count))
                    obj = json.loads(line.decode("utf-8"))
                    # if no exception is thrown, obj contains a record for a technote
                    try:
                        id = str(obj['example_id'])
                        if id in qrels:
                            title = obj['document_title']
                            doc_id = normalize_title(title)
                            tokens = obj['document_tokens']
                            text = ' '.join([token['token'] for token in tokens])

                            # convert to txt
                            soup = BeautifulSoup(text)
                            text = soup.get_text()
                            if self.document_mode:
                                yield record(doc_id, title=title, text=text, url=doc_id,
                                                collection=self.dataset_name, productName = doc_id)

                            # passage mode
                            else:
                                # for vdoc runs, add just the gold evidence as a record
                                if VDOC_MODE:
                                    passages = []
                                    for annotation in obj['annotations']:
                                        long_answer = annotation['long_answer']
                                        start_token = long_answer['start_token']
                                        end_token = long_answer['end_token']
                                        span_tokens = tokens[start_token+1:end_token-1] # ignore the <P> and </P>
                                        evidence = ' '.join([token['token'] for token in span_tokens])
                                        passages.append(evidence)

                                # for indexing
                                else:
                                    passages, _ = self.tokenizer_utils.segment(text,
                                                                       passage_token_limit=PASSAGE_LEN)

                                passage_id = 0
                                for passage in passages:
                                    pid = create_unique_pid(doc_id, passage_id)
                                    yield record(pid, url=create_unique_url(doc_id, passage_id),
                                                    title='', text=passage, collection=self.dataset_name,
                                                    productName=doc_id)
                                    passage_id += 1

                    except Exception as e:
                        print(e)
                        pass

    # default filtering - keep all records
    def filter_records(self, records):
        return records





