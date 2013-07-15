movieclassification
===================

Movie genre classification by plot text.

Abstract
--------

Classifies movies into genres according to plot summary texts.

A movie classifier (e.g. `NaiveBayesClassifier`), once trained, estimates the most-likely genres a movie belongs to, according to past examples.

Method
------

We implement supervised-learning techniques for multi-category classification of 

### Feature Extraction ###

**Movie Data**: Each movie is stored in a JSON entry containing fields: 'name', 'genres', 'year' and 'plot'. We extract features only from the 'plot' field. Each feature (term) is defined as a unigram (word) which is longer than K characters. Each movie is stored into a `Movie` class (which inherits from `Document`). We only use the 'genres' and 'plot' fields in later methods ('name' is used for convenience methods only).

**Document Features Representation**: Each `Document` implements a "bag of words" representation of the original text: for each term in the document we remember the number of times it has appeared.

**Document Corpus**: We aggregate multiple `Document`s into a `DocumentCorpus`. For each term, the document corpus stores the number of times it has appeared in the entire corpus (`term_freq`), the number of documents it has appeared in (`doc_freq`), and the total count of terms and documents it represents. Aggregating data into document corpuses allows us to represent a training set more as a sparse vector.

**Category Corpuses**: 

### Classification ###


### Testing ###



Classification Metrics
----------------------

The following metrics are currently evaluated per-category:

* True Positives Count: `TP`
* True Negatives Count: `TN` 
* False Positives Count: `FP`
* False Negatives Count: `FN`
* Precision: `PREC = TP / (TP + FP)`
* Recall: `RCL = TP / (TP + FN)`
* Accuracy: `ACC = (TP+TN) / (TP+FP+TN+FN)`

Further defintions of these metrics is available at the relevant [wikipedia article](http://en.wikipedia.org/wiki/Precision_and_recall).

Usage
-----

The program is not designed to work from command-line at the moment, but its various components may be used by `import`ing in python. An example of a complete test loop (load, train, test) is provided in `movies.test(...)`.

### Load movie data ###

    from movies import load_all_movies
    movie_lists = load_all_movies(movie_files=['train.json', 'test.json'])
    training_movies = movie_lists[0]
    test_movies = movie_lists[1]

### Build a MultiCategoryCorpus from movies ###

    from text_classification import MultiCategoryCorpus
    corpus = MultiCategoryCorpus.build(mtrain)

### Create and train a NaiveBayesClassifier ###

    from text_classification import NaiveBayesClassifier
    classifier = NaiveBayesClassifier(corpus)
    stats = classifier.train(training_movies)

### Test classifier on list of movies ###

    stats = classifier.test(test_movies)

Requirements
------------

* Python 2.7+
* No libraries are required

### Optional Modifications ###

* Any NLTK stemmer from `nltk.stem` may replace `text_extraction.Stemmer` (via the `stem(word)` method)
* NLTK Stopwords from `nltk.corpus.stopwords` may be used to initialize the stopwords list in `text_extraction.UnigramCounter`.

Software Design
---------------

The following list describes the module structure inside the library, and the prime classes inside each of them:

* **text_extraction.py**: Text feature extraction logic.
    * `Stemmer` interface: for text stemming, compatible with `nltk.stem`'s `stem(word)` method.
    * `TextExtractor` interface: parses a text block and returns a histogram (count of terms appearances in the text).
    * `UnigramCounter` class: implements `TextExtractor`. Splits a text block into unigrams (words). Supports stopwords filtering, stemming and minimal word length.
* **text_classification.py**: Text classification logic.
    * `Document` class: bag-of-words representation of a document (i.e. movie data).
    * `DocumentCorpus` class: aggregates metrics over many `Document` classes.
    * `MultiCategoryCorpus` class: aggregates metrics of many categories using `DocumentCorpus` classes. 
    * `MultiCategoryClassifier` class: base class for multiple category classification. 
    * `NaiveBayesClassifier` class: concrete implementation of `MultiCategoryClassifier`, implementing the naive bayes technique described previously in this document.
    * `ClassificationStats` class: utility to store and display statistics about the classification of many movies, according to the metrics described previously in this document.
* **utils.py**: 
    * `PickleCache` class: base class for seamless persistent storage of objects to disk using pickle. Concrete classes must implement the `load_object` method, which parses the original file format into a python object.
    * `random_json_file_subset` function: can be used to create a random movie subsets out of training or test files.
* **movies.py**: Wrappers and methods for working with movies (not just generic 'documents'). Responsible for data parsing and I/O.
    * `Movie` class: wraps `Document` and provides convenience and I/O methods specific to movies and our datasets.
    * `MovieCache` class: concrete implementation of `PickleCache` for `Movie` lists. 


Evaluation on given datasets
----------------------------


Suggested Improvements
----------------------

### Algorithmic / Classification ###
* Handle variance in per-genre corpus size
* Experiment with other classification methods and metrics (TF-IDF, SVM, etc)
* Improved text cleanup (dimensionality reduction) using stopwords and stemming
* Movie-specific optimizations to corpus and classifier (manually tweak weights of words and categories according to data).
* Include NGram terms (e.g. Bigrams and Trigrams in addition to Unigrams). Employing NGram terms might give us higher quality data, such as actor names, locations, etc.

### Performance ###
* Training while loading the data: will allow the program to go over the training file just once, and work fully with generators without keeping all the movies in memory at once.
* Corpus and Classifier pickling/caching
* Use standard implementations for learning and language processing.Specifically "scikit-learn" and "nltk"
* Parallelize training/building procedures (e.g. using multiple threads/processes)

### Interface ###
* Full-bown command-line interface using `argparse`.
* Visualizations using `mathplotlib` or others.







Movie classification by plot text.

This program attempts to classify a movie into genres according to its' plot text. The program is currently incomplete, since the algorithm hasn't been implemented yet, only some of the tools required to analyze text blocks (movie plots) in bulk.

The idea is to create a (binary) classifier for each genre, providing an estimation (in percentages) on the likelihood of the current movie belonging to the genre. 

Two classes collect data on movies (training sets or tests):
PlotAnalysis - essentially a class implementing a "bag of words" analysis on movie plot, and accumulating such representations for more than one movie plot. Stores a histogram of each word's occurences in the set of movie plots, each word's frequency throughout all plots, and the number of movie plots collected so far which contain this word.
MoviesData - a container for a variety of PlotAnalysis classes, stored globally (for all words in all movie plots), and per movie-genre in the movie set. Once the MoviesData is loaded from the training corpus, we have a bunch of histograms (global or per-genre). We then filter these histograms according to statistical properties of the words: How many movies has this word appeared in (absolute or percentage)? 
The above classes are pickle'd to/from files to avoid re-analyzing the same data between tests/iterations.

The machine learning should consider individual words as features in the input text, accumulate a score for all words in the text block for the specified genre, and measure the error in the estimation. 

Eventually, we would like to optimize (find the minimum cost) each classifier according to our training set data.

A few (key) questions remain:
- Which distance metric (cost function) should be used for each word
- How to effectively accumulate the cost for all words in a text block, to come up with a probability for the entire text belonging to the genre
- If each word is considered a feature, how do we handle a very large number (thousands or more) of features? Maybe SVMs can do the trick, but not sure how exactly.
- The correct optimization method (minimize the cost function via gradient descent, use SVM related stuff, etc)



## Classifiers ##
Our objective is to create a binary classifier for each movie genre (tag), predicting whether or not a movie plot belongs to a certain genre or not. This classifier should predict its results with a degree of certainty / probability (value in [0,1] for 'belongs to genre' and for 'not belongs to genre').

## Features ##

Our text corpus consists of movie plots tagged with their genres. We wish to extract unigrams and bigrams from the movie plot texts.
For each unigram or bigram in the feature vector, we wish to experiment with the following features:
* TF-IDF

