import re

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')


class SentCleaner:
    def __init__(self, sent, conf):
        self.sent = sent
        self.sent_tokenized = None
        self.conf = conf

    def lower_case(self):
        if self.conf['lower']:
            self.sent = self.sent.lower()
        return self

    def tokenize(self):
        if self.conf['token']:
            self.sent_tokenized = word_tokenize(self.sent)
        return self

    def remove_stopwords(self, stop_words):
        if self.conf['remove_stop']:
            self.sent_tokenized = [word for word in self.sent_tokenized if word not in stop_words]
        return self

    def remove_punct(self):
        if self.conf['remove_punc']:
            self.sent_tokenized = [word for word in self.sent_tokenized if re.search('[a-z]', word)]
        return self

    def stem_words(self):
        if self.conf['stem']:
            stemmer = PorterStemmer()
            self.sent_tokenized = [stemmer.stem(word) for word in self.sent_tokenized]
        return self

    def remove_escapes(self):
        if self.conf['remove_esc']:
            stripped = [word.replace('\n', '') for word in self.sent_tokenized]
            self.sent_tokenized = [word for word in stripped if word != '']
        return self

    def clean_sent(self):
        self.lower_case() \
            .tokenize() \
            .remove_punct() \
            .remove_escapes() \
            .stem_words() \
            .remove_stopwords(stopwords.words('english'))

        return self.sent_tokenized

    def sent_v(self):
        return set(self.sent_tokenized)
