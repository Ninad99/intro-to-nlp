import random
import pickle
import nltk

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf


pos_reviews = open("positive.txt", "r").read()
neg_reviews = open("negative.txt", "r").read()

documents = []
all_words = []

# j is adjective, r is adverb, and v is verb
allowed_word_types = ["J"]

for review in pos_reviews.split("\n"):
    documents.append((review, "pos"))  # add it as positive review doc
    words = word_tokenize(review)  # tokenize words
    pos = pos_tag(words)  # part of speech tagging of words in review
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for review in neg_reviews.split("\n"):
    documents.append((review, "neg"))  # add it as negative review doc
    words = word_tokenize(review)  # tokenize words
    pos = pos_tag(words)  # part of speech tagging of words in review
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_docs = open("documents.pickle", "wb")
pickle.dump(documents, save_docs)
save_docs.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

training_set = featuresets[:9000]
testing_set = featuresets[9000:]

# train naive bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:",
      (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# save trained classifier
save_classifier = open("pickled_algorithms/naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train MNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:",
      (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# save trained classifier
save_classifier = open("pickled_algorithms/MNB_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# save trained classifier
save_classifier = open(
    "pickled_algorithms/BernoulliNB_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# save
save_classifier = open(
    "pickled_algorithms/LogisticRegression_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# save
save_classifier = open(
    "pickled_algorithms/SGDClassifier_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# save
save_classifier = open("pickled_algorithms/LinearSVC_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# train
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:",
      (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# save
save_classifier = open("pickled_algorithms/NuSVC_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(
    classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier,
    SGDClassifier_classifier,
    LinearSVC_classifier,
    NuSVC_classifier)

print("voted_classifier accuracy percent:",
      (nltk.classify.accuracy(voted_classifier, testing_set))*100)
