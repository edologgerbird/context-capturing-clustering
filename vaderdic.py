import gensim.models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import spatial
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from reading https://radimrehurek.com/gensim/models/word2vec.html
# training the w2v model from twitter dataset: NEED the cleaned data

corpus = [['yes', 'i', 'like', 'McDonalds'],
          ['the', 'movie', 'spiderman', "is", "great"]]
# this is just an example
# the actual corpus will be a list of words in sentences, that is cleaned from the twitter dataset


# uninitialised model
model = gensim.models.Word2Vec(
    vector_size=2,  # dimension of word vector, default 100
    window=5,  # window size, default 5
    min_count=1,  # ignores all wprds with total freq lower than this threshold, default 5
    alpha=0.001,  # initial learning rate, default 0.025
    negative=5,  # number of negative samples used, default 5, 0 if not intended
    sg=1,  # 1 for skip-gram, 0 for CBOW, default 0
    hs=1,  # 1 for hierarchical softmax, default 0
)

# build vocab
model.build_vocab(corpus, progress_per=10000)

# training
model.train(corpus, total_examples=model.corpus_count, epochs=30, report_delay=1)

model.save("word2vec.model")
# if finished training, to make model much more memory efficient:
# model.init_sims(replace=True)

# use examples
# get word vectors
print(model.wv['yes'])
print(model.wv.most_similar('yes', topn=3))  # n most similar words, positive = [], negative = []
print(model.wv.similarity('yes', 'like'))  # similarity between 2 words of choice
print(model.wv.doesnt_match("yes great movie".split()))  # which of the given words does not go with the others?

# visualizations: t-SNE reduce word vectors to lower dimension
sns.set_style("darkgrid")


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=50).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))


# tsnescatterplot(model,
#                 'movie',
#                 [i[0] for i in model.wv.most_similar(negative=["movie"])])

vocab = model.wv.index_to_key
print(len(vocab))  # a plain list of the key (word) in each index position

# obtaining Vader Sentiment lexicons
analyzer = SentimentIntensityAnalyzer()
lexicons = analyzer.lexicon  # total: 7506
emojis = analyzer.emojis  # total: 3570

# filter lexicons to keep those that also appear in vocab
lexicons_filtered = {k: v for k, v in lexicons.items() if k in vocab}
print(len(lexicons_filtered))

# convert to word vectors and initialize as centroids
centroids = []
for i in lexicons_filtered.keys():
    lexicon_wv = model.wv[i]
    centroids.append(lexicon_wv)
print(centroids)
# cosine similarity between vocab words and centroids
# cosine_similarity = 1 - spatial.distance.cosine(dataSetI, dataSetII)

# get the closest centroid for each word in vocab and its valence


# --- examples for VaderSentiment-------
sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
             "VADER is smart, handsome, and funny!",
             # punctuation emphasis handled correctly (sentiment intensity adjusted)
             "VADER is very smart, handsome, and funny.",
             # booster words handled correctly (sentiment intensity adjusted)
             "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
             "VADER is VERY SMART, handsome, and FUNNY!!!",
             # combination of signals - VADER appropriately adjusts intensity
             "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
             # booster words & punctuation make this close to ceiling for score
             "VADER is not smart, handsome, nor funny.",  # negation sentence example
             "The book was good.",  # positive sentence
             "At least it isn't a horrible book.",  # negated negative sentence with contraction
             "The book was only kind of good.",  # qualified positive sentence is handled correctly (intensity adjusted)
             "The plot was good, but the characters are uncompelling and the dialog is not great.",
             # mixed negation sentence
             "Today SUX!",  # negative slang with capitalization emphasis
             "Today only kinda sux! But I'll get by, lol",
             # mixed sentiment example with slang and constrastive conjunction "but"
             "Make sure you :) or :D today!",  # emoticons handled
             "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # emojis handled
             "Not bad at all"  # Capitalized negation
             ]
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
