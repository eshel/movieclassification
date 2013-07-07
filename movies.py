import json
import re
import copy
import cPickle

DEFAULT_TRAINING_FILE = 'movies_2000.json'
DEFAULT_TEST_FILE = 'movies_test.json'


BOW_PARAMS = {
    'word_min_length': 3,
}

WORD_FILTER_PARAMS = {
    'movie_max_freq': 0.1,
    'movie_min_count': 3,
}


class PlotAnalysis:
    def __init__(self, text='', params=BOW_PARAMS):
        self.params = params
        self.clear()
        self.analyze(text)

    def __str__(self):
        return "PlotAnalysis: %d unique, %d total in %d movies" % (self.unique_words(), self.total_count, self.movies_count)

    def clear(self):
        self._is_normal = False
        self.total_count = 0
        self.count = {}
        self.frequency = {}
        self.movies_count_for_word = {}
        self.movies_count = 0

    def del_word(self, w):
        if w in self.count:
            self._is_normal = False
            self.total_count -= self.count[w]
            del self.count[w]
            del self.movies_count_for_word[w]
            if w in self.frequency:
                del self.frequency[w]

    def _add_word_count(self, word, count=1):
        self._is_normal = False        
        if not word in self.count:
            self.count[word] = 0
        self.count[word] += count
        self.total_count += count

    def is_valid_word(self, w):
        min_length = self.params['word_min_length']
        return (len(w) >= min_length)

    def is_empty(self):
        return self.total_count == 0

    def _update_total(self):
        self._is_normal = False
        t = 0
        for w in self.count:
            t += self.count[w]
        self.total_count = t

    def _count_words_in_plot_text(self, plot_text):
        text = plot_text.lower()
        words = re.findall(r"[\w']+", text)
        done_words_in_text = set()
        self.movies_count += 1
        for w in words:
            if not w in done_words_in_text:
                done_words_in_text.add(w)
                if not w in self.movies_count_for_word:
                    self.movies_count_for_word[w] = 1
                else:
                    self.movies_count_for_word[w] += 1
            if self.is_valid_word(w):
                self._add_word_count(w, 1)
        self.frequency = {}

    def analyze(self, text):
        self._count_words_in_plot_text(text)
        self._normalize()

    def _normalize(self):
        self.frequency = {}
        if self.total_count > 0:
            total = float(self.total_count)
            for w in self.count:
                self.frequency[w] = float(self.count[w]) / total
        self._is_normal = True

    def unique_words(self):
        return len(self.count)

    def add(self, other_bag):
        self.movies_count += other_bag.movies_count
        for w in other_bag.count:
            if self.is_valid_word(w):
                self._add_word_count(w, other_bag.count[w])
                if not w in self.movies_count_for_word:
                    self.movies_count_for_word[w] = other_bag.movies_count_for_word[w]
                else:
                    self.movies_count_for_word[w] += other_bag.movies_count_for_word[w]

    def word(self, w, full_stats=False):
        res = {}
        if full_stats and (not self._is_normal):
            self._normalize()
        if w in self.count:
            res['count'] = self.count[w]
            res['movies'] = self.movies_count_for_word[w]
            res['movies_freq'] = float(res['movies']) / float(self.movies_count)
            if full_stats:
                res['freq'] = self.frequency[w]
        return res

    def clone(self):
        return copy.deepcopy(self)

    def words(self):
        return self.count.keys()

    def sorted_words(self):
        words = sorted(self.count.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in words]

    def sorted_words_by_movies(self):
        words = sorted(self.movies_count_for_word.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in words]

    def has_word(self, w):
        if w in self.count:
            return True
        else:
            return False


class MoviesData:    
    def __init__(self):
        self.clear()

    def __str__(self):
        return "MoviesData for %d movies in %d genres" % (self.movies_count, self.genres_count())

    def clear(self):
        self.movies_count = 0
        self.genres = {}
        self.words = PlotAnalysis()

    def genres_count(self):
        return len(self.genres)

    def _add_movie_plot(self, plot, genres):
        genres = [g.lower() for g in genres]
        plot_words = PlotAnalysis(plot)
        for g in genres:
            if not g in self.genres:
                self.genres[g] = {'movies_count': 1, 'words': plot_words}
            else:
                self.genres[g]['words'].add(plot_words)
                self.genres[g]['movies_count'] += 1
        self.words.add(plot_words)
        self.movies_count += 1

    def add_movie(self, movie_data):
        m = movie_data
        plot = m['plot']
        genres = [g.lower() for g in m['genres']]
        self._add_movie_plot(plot, genres)

    def analyze_movies_file(self, infile, add_to_current=False):
        if not add_to_current:
            self.clear()
        with open(infile, 'r') as f:
            for line in f:
                movie = json.loads(line)
                self.add_movie(movie)

    def del_word(self, w):
        self.words.del_word(w)
        for g in self.genres:
            self.genres[g]['words'].del_word(w)

    def word_genre_count(self, w):        
        if not self.words.has_word(w):
            return 0
        else:
            gcnt = 0
            for g in self.genres:
                if self.genres[g].has_word(w):
                    gcnt += 1
            return gcnt

    def word_genre_freq(self, w):
        return float(self.word_genre_count(w)) / float(self.genres_count())

    def filter_words(self, params=WORD_FILTER_PARAMS):
        deleted = 0        
        movie_max_freq = params['movie_max_freq']
        movie_min_count = params['movie_min_count']
        words = self.words.words()
        for (idx, w) in enumerate(words):
            rem = False
            stats = self.words.word(w)
            if stats['movies_freq'] > movie_max_freq:
                rem = True
            if stats['movies'] < movie_min_count:
                rem = True
            if rem:
                self.del_word(w)
                deleted += 1
        return deleted


def md_load(infile='md_filtered.pickle'):
    md = None
    with open(infile, 'r') as f:
        md = cPickle.load(f)
    return md


def md_save(md, outfile='md_filtered.pickle'):
    with open(outfile, 'w') as f:
        cPickle.dump(md, f)

