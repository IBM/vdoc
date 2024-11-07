import json
import re

class record:

    def __init__(self, id, productName=None, docType=None, url=None, title=None, title_paraphrases=None, text=None, content=None, collection=None):
        self._id = id
        self._productName=productName
        self._docType=docType
        self._url = url
        self._title = title
        self._title_paraphrases = title_paraphrases
        self._text = text
        self._content = content
        self._collection = collection

    @property
    def id(self):
        return self._id

    @property
    def productName(self):
        return self._productName

    @property
    def docType(self):
        return self._docType

    @property
    def url(self):
        return self._url

    @property
    def title(self):
        return self._title

    @property
    def title_paraphrases(self):
        return self._title_paraphrases

    @property
    def text(self):
        return self._text

    @property
    def content(self):
        return self._content

    @property
    def collection(self):
        return self._collection

    @property
    def passages(self):
        return None

    def set_title(self, title):
        self._title = title

    def set_title_paraphrases(self, title_paraphrases):
        self._title_paraphrases = title_paraphrases

    def toJson(self):
        clean_title = clean_txt(self.title)
        #clean_paraphrases = clean_txt(self.title_paraphrases)
        clean_text = clean_txt(self.text)
        doc = json.dumps({
            "id": self.id,
            "collection": self.collection,
            "url": self.url,
            "title": clean_title,
            #"title_paraphrases": clean_paraphrases,
            #"title_bigrams": clean_title,
            "productId": self.productName,
            "text": clean_text,
            #"text_bigrams": clean_text,
            #"title_and_text": clean_title + '. ' + clean_text
        })
        return doc

    def title_and_text(self):
        clean_title = clean_txt(self.title)
        clean_text = clean_txt(self.text)
        return f'{clean_title}. {clean_text}'


def clean_txt(txt):
    #txt = re.sub(r'([^\s\w,\\.]|_)+', ' ', txt).rstrip()
    txt = remove_non_unicode_chars(txt)
    #txt = remove_single_character_words(txt)
    return txt

def remove_non_unicode_chars(string):
    return ''.join([i if ord(i) < 128 else ' ' for i in string])

def remove_single_character_words(string):
    return ' '.join( [w for w in string.split() if len(w)>1] )
