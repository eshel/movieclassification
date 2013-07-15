import json

from text_classification import Document, MultiCategoryCorpus, NaiveBayesClassifier
from text_extraction import UnigramCounter
from utils import PickleCache

DEFAULT_TRAINING_FILE = 'movies_train.json'
DEFAULT_TEST_FILE = 'movies_test.json'

DEFAULT_TOKENIZER = UnigramCounter()
DEFAULT_CLASSIFIER = NaiveBayesClassifier


class Movie(Document):
    def __init__(self, movie_json, tokenizer=DEFAULT_TOKENIZER):
        title = movie_json['name']
        categories = [g.lower() for g in movie_json['genres']]
        plot = movie_json['plot']
        term_freq = tokenizer.parse(plot)
        super(Movie, self).__init__(title=title, categories=categories, term_freq=term_freq)

    def _get_genres(self):
        return self.categories
    genres = property(_get_genres)

    def _get_plot(self):
        return self.text

    def is_in_genre(self, genre):
        for g in self.genres:
            if g == genre:
                return True
        return False

    @staticmethod
    def movies_in_file(f):
        for line in f:
            movie = Movie(json.loads(line))
            yield movie

    @staticmethod
    def all_movies_in_file(infile):
        with open(infile, 'r') as f:
            return list(Movie.movies_in_file(f))


class MovieCache(PickleCache):
    def __init__(self, pickle_suffix='.pickle'):
        super(MovieCache, self).__init__(pickle_suffix=pickle_suffix)

    def load_object(self, fpath):
        return Movie.all_movies_in_file(fpath)


def load_all_movies(movie_files=[DEFAULT_TRAINING_FILE, DEFAULT_TEST_FILE], clear_cache=False):    
    cache = MovieCache()
    return [cache.read(mf, clear_cache) for mf in movie_files]


def test(clear_cache=False, classifier_class=DEFAULT_CLASSIFIER, verbose=True):
    """Loads the training and test datasets, then trains and tests it."""

    if verbose: print('loading movies')
    M = load_all_movies(movie_files=[DEFAULT_TRAINING_FILE, DEFAULT_TEST_FILE], clear_cache=clear_cache)
    (mtrain, mtest) = (M[0], M[1])

    if verbose: print('building corpus')
    corpus = MultiCategoryCorpus.build(mtrain)

    if verbose: print('training classifiers')
    classifier = classifier_class(corpus)
    train_stats = classifier.train(mtrain)

    if verbose: print('testing classification')
    test_stats = classifier.test(mtest)
    results = {'mtrain': mtrain, 
               'mtest': mtest, 
               'corpus': corpus, 
               'classifier': classifier, 
               'strain': train_stats,
               'stest': test_stats,
               }

    if verbose: test_stats.print_stats()
    return results
