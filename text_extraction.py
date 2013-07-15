import re
from collections import defaultdict

#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer


class Stemmer(object):
    """Base class for word stemmers."""
    def __init__(self):
        pass

    def stem(self, word):
        return word


class TextExtractor(object):
    """Base class for text extraction"""
    def __init__(self):
        pass

    def parse(self, document):
        """Parses the text in document, returns a histogram (count of terms)"""
        raise NotImplemented()


class UnigramCounter(TextExtractor):
    def __init__(self, 
                 word_min_length=2, 
                 stopwords=[],
                 stemmer=Stemmer(),
                 ):
        super(UnigramCounter, self).__init__()
        word_re = (r'\w' * word_min_length) + r'+'
        self.word_splitter = re.compile(word_re)
        self.stopwords = set(stopwords)
        self.stemmer = stemmer

    def split_words(self, document):
        return self.word_splitter.findall(document.lower())

    def parse(self, document):
        words = self.split_words(document)
        histogram = defaultdict(int)
        for w in words:
            if not self.stemmer is None:
                w = self.stemmer.stem(w)
            if not w in self.stopwords:
                histogram[w] += 1
        return histogram
