movieclassification
===================

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
