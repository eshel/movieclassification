from collections import defaultdict
import math


def _div(a, b):
    return (float(a) / b) if b else 0


class Document(object):
    term_freq = defaultdict(int)
    terms_count = 0
    max_freq = 0
    categories = []
    title = ''

    def __init__(self, title='', categories=[], term_freq=defaultdict(int)):
        self.title = title
        self.categories = categories
        self.term_freq = term_freq
        self.terms_count = 0
        self.max_freq = 0
        for (term, count) in self.term_freq.items():
            self.terms_count += count
            self.max_freq = max(self.max_freq, count)

    def __str__(self):
        doc_title = self.title
        if len(doc_title) == 0:
            doc_title = self.__class__.__name__
        return "%s: %d unique terms (%d total), %d categories" % (doc_title, len(self.term_freq), self.terms_count, len(self.categories))

    def _get_terms(self):
        return self.term_freq.keys()
    terms = property(_get_terms)

    def in_category(self, cat):
        return cat in self.categories

    def tf(self, term):
        if term in self.term_freq:
            return float(self.term_freq[term]) / float(self.max_freq)
        else:
            return 0


class DocumentCorpus(object):
    term_freq = defaultdict(int)
    doc_freq = defaultdict(int)
    docs_count = 0
    terms_count = 0

    def __init__(self):
        self.clear()

    def __str__(self):
        return "Corpus of %d docs with %d unique terms" % (self.docs_count, self.unique_terms_count)

    def _get_unique_terms_count(self):
        return len(self.term_freq)
    unique_terms_count = property(_get_unique_terms_count)

    def _get_terms(self):
        return self.term_freq.keys()
    terms = property(_get_terms)

    def clear(self):
        self.term_freq.clear()
        self.doc_freq.clear()
        self.docs_count = 0
        self.terms_count = 0

    def add(self, document):
        for (term, count) in document.term_freq.items():
            self.term_freq[term] += count
            self.doc_freq[term] += 1
        self.docs_count += 1
        self.terms_count += document.terms_count    


class MultiCategoryCorpus(object):
    def __init__(self):
        self.universe = DocumentCorpus()
        self.category_corpus = {}
        self.clear()

    def __str__(self):
        return "MultiCorpus of %d docs in %d categories, with %d unique terms" % (
            self.universe.docs_count, len(self.category_corpus), len(self.universe.term_freq))

    def clear(self):
        self.universe.clear()        
        self.category_corpus.clear()

    def add(self, document):        
        categories = document.categories
        self.universe.add(document)
        for c in categories:
            if not c in self.category_corpus:
                self.category_corpus[c] = DocumentCorpus()
            self.category_corpus[c].add(document)

    def _get_categories(self):
        return self.category_corpus.keys()
    categories = property(_get_categories)

    def category_weight(self, category):
        return float(self.category_corpus[category].docs_count) / self.universe.docs_count

    def weighted_categories(self):
        return dict((c, self.category_weight(c)) for c in self.categories)

    @staticmethod
    def build(documents):
        mcc = MultiCategoryCorpus()
        for doc in documents:
            mcc.add(doc)
        return mcc


class ClassificationStats(object):
    def __init__(self, categories):
        self.stats = {}
        self.total = 0
        self.categories = categories
        self.clear()

    def __str__(self):
        return "ClassificationStats for %d samples" % (self.total)

    def _get_category_weight(self, c):
        return float(self.stats[c]['ctp'] + self.stats[c]['cfn']) / self.total

    def _get_sorted_categories(self):
        weighted_categories = [(c, self._get_category_weight(c)) for c in self.categories]
        srt = sorted(weighted_categories, key=lambda x: x[1], reverse=True)
        return srt

    def clear(self):
        self.total = 0
        self.stats.clear()
        st = {'count': 0, 'sum': 0.0, 'min': 0.0, 'max': 0.0, 'avg': 0.0}
        for c in self.categories:            
            self.stats[c] = {'tp': st.copy(),    # true positive
                             'fp': st.copy(),    # false positive
                             'fn': st.copy(),    # false negative
                             'tn': st.copy(),    # true negative
                             'ctp': 0,
                             'cfp': 0,
                             'ctn': 0,
                             'cfn': 0,
                             }

    def _add_score(self, c, k, score):
        self.stats[c]['c' + k] += 1
        if not score is None:
            is_first = self.stats[c][k]['count'] == 0
            self.stats[c][k]['count'] += 1
            self.stats[c][k]['sum'] += score
            self.stats[c][k]['avg'] = float(self.stats[c][k]['sum']) / self.stats[c][k]['count']        
            if is_first:
                self.stats[c][k]['min'] = score
                self.stats[c][k]['max'] = score
            else:
                self.stats[c][k]['min'] = min(self.stats[c][k]['min'], score)
                self.stats[c][k]['max'] = max(self.stats[c][k]['min'], score)

    def add(self, actual, decisions):
        actual = dict((cat, True) for cat in actual)
        positive_scores = dict((c, w) for (c, is_positive, w) in decisions if is_positive)
        negative_scores = dict((c, w) for (c, is_positive, w) in decisions if not is_positive)
        for c in self.categories:
            in_actual = c in actual
            in_predicted = c in positive_scores
            if in_predicted:
                score = positive_scores[c]
            else:
                score = negative_scores[c]

            if (in_actual) and (in_predicted):
                self._add_score(c, 'tp', score)
            elif (in_actual) and (not in_predicted):
                self._add_score(c, 'fn', score)
            elif (not in_actual) and (in_predicted):
                self._add_score(c, 'fp', score)
            else:
                self._add_score(c, 'tn', score)
        self.total += 1

    def category(self, category):
        s = self.stats[category]
        (tp, fp, tn, fn) = (s['ctp'], s['cfp'], s['ctn'], s['cfn'])
        s['positives'] = tp + fn
        s['negatives'] = tn + fp
        s['true'] = tp + tn
        s['false'] = fp + fn
        s['precision'] = _div(tp, tp + fp)
        s['recall'] = _div(tp, tp + fn)
        s['accuracy'] = _div(tp + tn, tp + fp + tn + fn)
        return s

    def totals(self):
        totals = {'ctp': 0, 'ctn': 0, 'cfp': 0, 'cfn': 0, 'positives': 0, 'negatives': 0}
        for c in self.categories:
            cs = self.category(c)
            for metric in totals:
                totals[metric] += cs[metric]
        return totals

    def averages(self):
        weights_sum = 0.0
        averages = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        for c in self.categories:
            cw = self._get_category_weight(c)
            cs = self.category(c)
            for metric in averages:
                averages[metric] += cw * cs[metric]
            weights_sum += cw
        for metric in averages:
            averages[metric] = averages[metric] / weights_sum
        return averages

    def print_stats(self):
        average_metrics = [('precision', 'PREC'), 
                           ('recall', 'RCL'), 
                           ('accuracy', 'ACC'),
                           ]

        total_metrics = [('positives', '#POS'),
                         ('ctp', 'TP'),
                         ('ctn', 'TN'),
                         ('cfp', 'FP'),
                         ('cfn', 'FN'),
                         ]

        # title
        line = '%-20s ' % ('CATEGORY')
        for m in average_metrics:
            line += '| %6s ' % m[1]
        for m in total_metrics:
            line += '| %5s ' % m[1]
        line += '|'
        print(line)

        # data
        data = [(c, cw, self.category(c)) for (c, cw) in self._get_sorted_categories()]
        total_stats = dict(self.averages().items() + self.totals().items())
        data.append(('TOTAL', 1.0, total_stats))
        for (c, cw, stats) in data:
            cstr = '%s: %1.2f%%' % (c, cw * 100.0)
            line = '%-20s ' % (cstr)
            for m in average_metrics:
                val = int(stats[m[0]] * 1000)
                line += '| %3d.%1d%% ' % (val / 10, val % 10)
            for m in total_metrics:
                val = stats[m[0]]
                line += '| %5d ' % (val)
            line += '|'
            print(line)


class MultiCategoryClassifier(object):
    def __init__(self, multi_category_corpus):
        self.mcc = multi_category_corpus
        self.thresholds = {}
        self.training_stats = None
        for c in self.categories:
            self.thresholds[c] = None

    def _get_categories(self):
        return self.mcc.category_corpus.keys()
    categories = property(_get_categories)

    def classify_one_category(self, document, category):
        category_corpus = self.mcc.category_corpus[category]
        universal_corpus = self.mcc.universe
        w = self.document_weight(document, category_corpus, universal_corpus)
        return (self.is_document_weight_positive(w, category), w)

    def classify_all(self, document):
        decisions = []
        for c in self.categories:
            (is_positive, w) = self.classify_one_category(document, c)
            decisions.append((c, is_positive, w))
        return decisions

    def decisions_to_positives(self, decisions):
        return [c for (c, is_positive, w) in decisions if is_positive]

    def is_document_weight_positive(self, weight, category):
        threshold = self.thresholds[category]
        if not threshold is None:
            return weight > threshold
        else:
            return True

    def classify(self, document):
        decisions = self.classify_all(document)
        return MultiCategoryClassifier.sort_decisions(decisions)

    def test(self, documents):
        cs = ClassificationStats(self.mcc.categories)
        for doc in documents:
            actual = doc.categories
            decisions = self.classify_all(doc)
            cs.add(actual, decisions)
        return cs

    def train(self, documents):
        cs = ClassificationStats(self.mcc.categories)
        for doc in documents:
            actual = doc.categories
            decisions = self.classify_all(doc)
            cs.add(actual, decisions)
        for c in self.categories:
            fp_max = cs.stats[c]['fp']['max']
            self.thresholds[c] = fp_max
        self.training_stats = cs
        return cs

    @staticmethod
    def sort_decisions(decisions):
        srt = sorted(decisions, key=lambda x: x[2], reverse=True)
        return srt

    def term_weight(self, term, category_corpus, universal_corpus):
        raise NotImplementedError()

    def document_weight(self, document, category_corpus, universal_corpus):
        raise NotImplementedError()


class NaiveBayesClassifier(MultiCategoryClassifier):
    def __init__(self, multi_category_corpus, max_categories=5, percentile=0.2):
        super(NaiveBayesClassifier, self).__init__(multi_category_corpus)
        self.max_categories = max_categories
        self.percentile = percentile

    def weighted_prob(self, term, category_corpus, universal_corpus, use_positive=True, weight=1.0, ap=0.5):
        cc = category_corpus
        uc = universal_corpus
        if use_positive:
            in_category = float(cc.term_freq[term]) / float(cc.terms_count)
        else:
            in_category = float(uc.term_freq[term] - cc.term_freq[term]) / float(cc.terms_count)
        in_all = uc.term_freq[term]
        wp = float((weight*ap) + (in_all*in_category)) / (weight+in_all)
        return wp

    def document_weight(self, document, category_corpus, universal_corpus):
        cc = category_corpus
        uc = universal_corpus
        category_prob = float(cc.docs_count) / uc.docs_count
        logsum = math.log(category_prob, 2)
        for (term, cnt) in document.term_freq.items():
            weight = self.weighted_prob(term, cc, uc)
            logsum += math.log(weight, 2)
        return logsum

    def filter_decisions(self, decisions):
        return decisions


"""
class TfidfClassifier(MultiCategoryClassifier):
    def __init__(self, multi_category_corpus, max_categories=5, percentile=0.2):
        super(TfidfClassifier, self).__init__(multi_category_corpus)
        self.max_categories = max_categories
        self.percentile = percentile

    def document_weight(self, document, category_corpus, universal_corpus):
        sum_tfidf = 0
        ccorpus = category_corpus
        for term in document.term_freq:
            idf = ccorpus.idf[term]
            tf = document.tf(term)
            tfidf = tf * idf
            sum_tfidf += tfidf
        return sum_tfidf / document.terms_count

    def filter_decisions(self, decisions):
        return decisions
"""
