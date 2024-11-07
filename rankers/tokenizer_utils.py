from typing import List

from transformers import AutoTokenizer
from nltk import tokenize


class TokenizerUtils:

    def __init__(self, tokenizer=None, model='google/flan-t5-small'):
        if tokenizer:
            self.tokenizer = tokenizer
        elif model:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    # computes the token length for the given text
    def get_length(self, text):
        return self.tokenizer(text, padding=False, return_length=True,
                              return_attention_mask=False)['length'][0]

    # computes the token length for the given list of texts, and also
    # the offset mapping between tokens and characters.
    def get_lengths(self, texts):
        tokenizer_result = self.tokenizer(texts, padding=False, return_length=True,
                                          return_attention_mask=False,
                                          return_offsets_mapping=True,
                                          add_special_tokens=False,
                                          return_special_tokens_mask=True)
        return tokenizer_result['length'], tokenizer_result['offset_mapping']

    # break long sentences by newline
    def extract_sentences(self, content, sentence_token_limit):
        # use nltk's sentence tokenizer to break the content into sentences
        # and use the length function to compute their length
        nltk_sentences = tokenize.sent_tokenize(content)
        nltk_lengths, _ = self.get_lengths(nltk_sentences)

        # sometimes we get very long sentences from NLTK sentence breaker,
        # so we break them further by newlines or by the token limit as last resort.

        # collecting the final lists of sentences and lengths
        sentences = []
        lengths = []
        for sentence, sentence_len in zip(nltk_sentences, nltk_lengths):
            if sentence_len > sentence_token_limit:
                # if the sentence is too long, try to split it by newline
                lines = sentence.split("\n")

                # get rid of empty lines
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if len(line) > 0]

                lines_length = self.get_lengths(lines)
                for line, line_len, token_spans in zip(lines, lines_length[0], lines_length[1]):
                    if line_len > sentence_token_limit:
                        # if it's still too long, split by the token limit
                        sub_lines_token_span = [(i, min(i + sentence_token_limit, line_len))
                                                for i in range(0, line_len, sentence_token_limit)]
                        for start_token, end_token in sub_lines_token_span:
                            sentences.append(line[token_spans[start_token][0]:
                                                  token_spans[end_token - 1][1]])
                            lengths.append(end_token - start_token)
                    else:
                        sentences.append(line)
                        lengths.append(line_len)
            else:
                sentences.append(sentence)
                lengths.append(sentence_len)

        return sentences, lengths

    # segment to fixed window size (use sentence boundaries)
    def segment(self, content, passage_token_limit):
        sentences, lengths = self.extract_sentences(content, passage_token_limit)

        passages = []
        passages_length = []
        passage_length = 0
        passage_sentences = list()
        for sentence, sentence_len in zip(sentences, lengths):
            if passage_length + sentence_len <= passage_token_limit:
                passage_sentences.append(sentence)
                passage_length += sentence_len
            else:
                passage = ' '.join(passage_sentences)
                passages.append(passage)
                passages_length.append(passage_length)

                passage_length = sentence_len
                passage_sentences = [sentence]

        # add the last passage
        if len(passage_sentences) > 0:
            passage = ' '.join(passage_sentences)
            passages.append(passage)
            passages_length.append(passage_length)

        return passages, passages_length

    # Collect texts up to a given maximal token count,
    # with optional limit on the number of texts.
    # It uses a tokenizer to compute the token count.
    def collect(self, texts: List[str], max_length: int,
                lengths: List[int] = None,
                max_texts: int = None):

        # slice up to max_texts (if specified)
        if max_texts:
            texts = texts[0:max_texts]

            if lengths is not None:
                lengths = lengths[0:max_texts]

        # if lengths are not provided, compute them using the
        # provided tokenizer
        if lengths is None:
            lengths = self.tokenizer(texts, padding=False,
                                     return_length=True,
                                     return_attention_mask=False,
                                     add_special_tokens=False)['length']

        # accumulate up to the max_length
        total_length = 0
        total_texts = 0
        for text, length in zip(texts, lengths):
            if total_length + length < max_length:
                total_length += length
                total_texts += 1
            else:
                break
        texts = texts[0:total_texts]

        return texts

    # Collect texts up to a given maximal token count,
    # with optional limit on the number of texts,
    # and then joins the texts into a single string
    # It uses a tokenizer to compute the token count.
    def collect_and_join(self, texts: List[str], max_length: int, lengths: List[int] = None,
                         max_texts: int = None, connector: str = '\n\n'):

        return connector.join(self.collect(texts, max_length, lengths, max_texts))

    # remove last sentence if it is truncated
    def remove_last_truncated_sentence(self, content):
        sentences = tokenize.sent_tokenize(content)
        if len(sentences) > 1:
            last_sentence = sentences[-1]
            if not last_sentence.endswith(('.','?','!')):
                last_sentence_offset = content.rfind(last_sentence)
                content = content[0:last_sentence_offset]
        return content

def main():
    texts = ['If your vehicle is stolen, report it to the police and your auto insurance company as soon as possible.',
             'The police will enter the information into national and state auto theft computer records.',
             'The theft will be noted on your vehicle title record to help prevent someone from selling the vehicle '
             'or applying for a title.']

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    utils = TokenizerUtils(tokenizer)

    print("Collect with max-length=150")
    print("[\n" + utils.collect_and_join(texts, max_length=150) + "\n]")
    print()

    print("Collect with max-length=150, max-texts=2")
    print("[\n" + utils.collect_and_join(texts, max_length=150, max_texts=2) + "\n]")
    print()

    lengths = tokenizer(texts, padding=False,
                        return_length=True,
                        return_attention_mask=False,
                        add_special_tokens=False)['length']

    utils = TokenizerUtils()
    print("Collect with max-length=150, max-texts=2, pre-given lengths (no tokenizer)")
    print("[\n" + utils.collect_and_join(texts, lengths=lengths, max_length=150, max_texts=2) + "\n]")
    print()

    print("Collect with max-length=150, max-texts=2, no tokenizer - should fail")
    try:
        print("[\n" + utils.collect_and_join(texts, max_length=150, max_texts=2) + "\n]")
    except ValueError as e:
        print(f"Expected exception raised: {e}")

    print()


if __name__ == '__main__':
    main()
