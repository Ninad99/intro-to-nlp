import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# documents = [(list(movie_reviews.words(fileid)), category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]


documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)
all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# word_features contains the top 4,000 most common words
word_features = list(all_words.keys())[:4000]

# find_features() will find these top 4,000 words in the positive and negative
# documents, marking their presence as either positive or negative


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# training
# classifier = nltk.NaiveBayesClassifier.train(training_set)

# for saving the classifier after training
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

print("Classifier accuracy percent: ",
      (nltk.classify.accuracy(classifier, testing_set)) * 100)

classifier.show_most_informative_features(20)
