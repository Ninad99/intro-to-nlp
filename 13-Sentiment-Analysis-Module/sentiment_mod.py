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


documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

featuresets_f = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

# Original Naive Bayes
open_file = open("pickled_algorithms/naivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

# MNB
open_file = open("pickled_algorithms/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

# BernoulliNB
open_file = open("pickled_algorithms/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

# Logistic Regression
open_file = open(
    "pickled_algorithms/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

# SGDClassifier
open_file = open("pickled_algorithms/SGDClassifier_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

# LinearSVC
open_file = open("pickled_algorithms/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

# NuSVC
open_file = open("pickled_algorithms/NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
    classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier,
    SGDClassifier_classifier,
    LinearSVC_classifier,
    NuSVC_classifier)


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
