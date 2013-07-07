import json
import re
import copy

HISTOGRAM_FILTER_PARAMS = {
    'word_min_length': 3,
    'word_freq': (0.0, 0.20),
    'word_genre_prob': (0.10, 1.00),
    'word_genre_spread': (1, 8),
    'movie_min_count': 3,
}

GENRE_FILTER_PARAMS = {
    'genre_prob_min': 0.15,
}


def hist_params_str(params):
    p = params
    return '{ genre_prob > %d%%, genre_spread <= %d, movies >= %d, word_freq in [%f,%f] }' % (
        int(p['word_genre_prob'][0] * 100), p['word_genre_spread'][1], p['movie_min_count'],
        p['word_freq'][0], p['word_freq'][1])


DEFAULT_TRAINING_FILE = 'movies_med.json'
DEFAULT_TEST_FILE = 'movies_test.json'
DEFAULT_RAW_HIST_FILE = 'hist_raw.json'


class MovieClassifier:
    words = {}
    genres = {}

    def __init__(self):
        pass

    def words_count(self):
        return len(self.words)

    def genres_count(self):
        return len(self.genres)

    def filter(self, params=HISTOGRAM_FILTER_PARAMS):
        filtered = 0
        for w in self.words.keys():
            freq = self.words[w]['dictfreq']
            spread = self.words[w]['genre_spread']
            moviescount = self.words[w]['movies']

            # Pre-process
            too_short = (len(w) < params['word_min_length'])
            dict_freq_ok = (params['word_freq'][0] <= freq <= params['word_freq'][1])
            word_in_enough_movies = (moviescount >= params['movie_min_count'])
            if too_short or not dict_freq_ok or not word_in_enough_movies:
                del self.words[w]
                filtered += 1
            else:
                for (g, p) in self.words[w]['genres'].items():
                    if not (params['word_genre_prob'][0] <= p <= params['word_genre_prob'][1]):
                        del self.words[w]['genres'][g]
                self.normalize_genres(w)
                if not (params['word_genre_spread'][0] <= spread <= params['word_genre_spread'][1]):
                    del self.words[w]
                    filtered += 1

        return filtered        

    def normalize_genres(self, word):
        w = word
        total = sum([self.words[w]['genres'][genre] for genre in self.words[w]['genres']])
        self.words[w]['genre_spread'] = len(self.words[w]['genres'])
        for (genre, count) in self.words[w]['genres'].items():
            self.words[w]['genres'][genre] = float(count) / float(total)        

    def train(self, training_file=DEFAULT_TRAINING_FILE, text_field='plot', filtered=True):
        self.words = {}
        total_words_count = 0
        movies_count = 0
        with open(training_file, 'r') as training_movies:
            for line in training_movies:
                movie = json.loads(line)
                movies_count += 1
                text_words = split_words(movie[text_field])
                seen_words_in_text = set()
                for w in text_words:
                    total_words_count += 1
                    if not w in self.words:
                        self.words[w] = {'count': 0, 'genres': {}, 'movies': 0}
                    self.words[w]['count'] += 1
                    if not w in seen_words_in_text:
                        self.words[w]['movies'] += 1
                        seen_words_in_text.add(w)
                    for genre in movie['genres']:
                        genre = genre.lower()
                        if not genre in self.words[w]['genres']:
                            self.words[w]['genres'][genre] = 0
                        self.words[w]['genres'][genre] += 1

        # frequency of word in dictionary
        for w in self.words:
            self.words[w]['dictfreq'] = float(self.words[w]['count']) / float(total_words_count)
            self.words[w]['moviefreq'] = float(self.words[w]['movies']) / float(movies_count)

        # genre normalization
        for w in self.words:
            self.normalize_genres(w)

        if filtered:
            self.filter()

    def save(self, outfile):
        with open(outfile, 'w') as f:
            f.write(json.dumps(self.words))

    def load(self, infile=DEFAULT_RAW_HIST_FILE):
        with open(infile, 'r') as f:
            self.words = json.loads(f.read())

    def duplicate(self):        
        return copy.deepcopy(self)

    def str_for_word(self, word):
        GENRES_NUM = 4
        prefix = '%-15s (%1.3f%% == %6d, %4d movies)' % (word, self.words[word]['dictfreq'] * 100, self.words[word]['count'], self.words[word]['movies'])
        genres = sorted(self.words[word]['genres'].items(), key=lambda x: x[1], reverse=True)
        s = prefix + ': '
        for (genre, prob) in genres[:GENRES_NUM]:
            s += '%s: %2.1f%%, ' % (genre, prob * 100)
        if len(genres) > GENRES_NUM:
            s += '...'
        return s

    def print_words(self):
        for kw in self.words:
            print(self.str_for_word(kw))

    def guess_movie_genres(self, movie, filtered=True):
        genre_scores = {}
        text_words = split_words(movie['plot'])
        word_hits = 0
        for w in text_words:
            if w in self.words:
                word_hits += 1
                for genre in self.words[w]['genres']:
                    score = genre_assign_score(self.words[w], genre)
                    if not genre in genre_scores:
                        genre_scores[genre] = 0
                    genre_scores[genre] += score
        normalize_genre_scores(genre_scores)

        if filtered:
            filter_genre_scores(genre_scores)
        return genre_scores

    def guess_all(self, movies):
        mstats = {'hits': 0.0, 'misses': 0.0, 'falses': 0.0}
        for m in movies:
            actual_genres_count = len(m['genres'])
            m['guesses'] = self.guess_movie_genres(m)
            all_genres = {}
            for g in m['genres']:
                all_genres[g] = {'actual': 1.0 / float(actual_genres_count), 'guessed': 0.0}
            for (g, prob) in m['guesses'].items():
                if not g in all_genres:
                    all_genres[g] = {'actual': 0.0}
                all_genres[g]['guessed'] = prob
            m['all_genres'] = all_genres

            stats = sort_all_genres(m)
            hits = 0
            misses = 0
            falses = 0
            count = 0
            EPSILON = 0.01

            for (genre, vals) in stats:
                actual = vals['actual']
                guessed = vals['guessed']
                count += 1
                if actual > EPSILON:
                    if guessed > EPSILON:
                        hits += 1
                    else:
                        misses += 1
                else:
                    falses += 1
            m['stats'] = {'hits': hits, 'misses': misses, 'falses': falses, 'total': count}
            m['stats_norm'] = {}
            for st in [k for k in m['stats'].keys() if k != 'total']:
                m['stats_norm'][st] = float(m['stats'][st]) / float(count)
                mstats[st] += m['stats_norm'][st]

        # Aggregated Stats for all movies
        mnum = len(movies)
        for k in mstats.keys():
            mstats[k] = mstats[k] / float(mnum)

        return mstats


def split_words(text):
    text = text.lower()
    words = re.findall(r"[\w']+", text)
    return words


def normalize_genre_scores(genre_scores):
    total = sum(genre_scores.values())
    for genre in genre_scores:
        genre_scores[genre] = genre_scores[genre] / total
    return genre_scores


def filter_genre_scores(genre_scores, params=GENRE_FILTER_PARAMS):
    for g in genre_scores.keys():
        if genre_scores[g] < params['genre_prob_min']:
            del genre_scores[g]
    normalize_genre_scores(genre_scores)
    return genre_scores


def genre_assign_score(word_entry, genre):    
    genre_likelihood_for_word = word_entry['genres'][genre]
    genre_spread = word_entry['genre_spread']
    movies_count = word_entry['movies']
    movies_count = word_entry['dictfreq']

    weight_genre_spread = 1/float(genre_spread)

    score = genre_likelihood_for_word * weight_genre_spread * movies_count
    return score


def load_test_movies(test_file=DEFAULT_TEST_FILE):
    movies = []
    with open(test_file, 'r') as f:
        for line in f:
            m = json.loads(line)
            m['genres'] = [g.lower() for g in m['genres']]
            movies.append(m)
    return movies


def sort_all_genres(movie):
    return sorted(movie['all_genres'].items(), key=lambda x: x[1]['guessed'], reverse=True)


def print_guess(movie, detailed=False):
    m = movie
    print('%-30s: %d Total, hit: %3d%%, miss: %3d%%, false: %3d%%' % 
        (m['name'][:30], m['stats']['total'], int(m['stats_norm']['hits']*100), int(m['stats_norm']['misses']*100), int(m['stats_norm']['falses']*100)))
    if detailed:
        for (g, vals) in sort_all_genres(m):
            print('\t%-16s: Guessed %3d%%, Actual %3d%%' % (g, int(vals['guessed'] * 100), int(vals['actual'] * 100)))


def create_raw_hist(training_file=DEFAULT_TRAINING_FILE, out_file=DEFAULT_RAW_HIST_FILE):
    mc = MovieClassifier()
    mc.train(training_file, filtered=False)
    mc.save(out_file)
    return mc


def run(trace=False):
    print('Loading data files...')
    mc_raw = MovieClassifier()
    mc_raw.load(DEFAULT_RAW_HIST_FILE)
    movies_original = load_test_movies()

    print('Permutating...')
    for max_genre_spread in [10]:
        for min_movies in [3]:
            for min_genre_prob in [5]:
                for min_word_freq in [0]:
                    for max_word_freq in [0.01]:
                        hist_filter_params = HISTOGRAM_FILTER_PARAMS
                        hist_filter_params['word_genre_spread'] = (1, max_genre_spread)
                        hist_filter_params['movie_min_count'] = min_movies
                        hist_filter_params['word_genre_prob'] = (float(min_genre_prob) / 100.0, 1.0)
                        hist_filter_params['word_freq'] = (min_word_freq, max_word_freq)

                        print('Copying')
                        mc = mc_raw.duplicate()
                        movies = copy.deepcopy(movies_original)
                        print('Guessing')
                        mc.filter(hist_filter_params)
                        mstats = mc.guess_all(movies)

                        if trace:
                            for m in movies:
                                print_guess(m)
                        print('For %d Movies: hit: %02.1f%%, miss: %02.1f%%, false: %02.1f%%' % (len(movies), 
                            mstats['hits'] * 100, 
                            mstats['misses'] * 100, 
                            mstats['falses'] * 100))
                        print('\t%s --> Histogram Size: %d' % (hist_params_str(hist_filter_params), mc.words_count())) 
    print('DONE')
