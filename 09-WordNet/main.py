# Full tutorial - https://pythonprogramming.net/wordnet-nltk-tutorial/

'''
WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus.

You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more.
'''

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# a single synset
print(syns[0].name())

# a list of lemmas
print(syns[0].lemmas())

# a single lemma - only the word
print(syns[0].lemmas()[0].name())

# the definition of a synset
print(syns[0].definition())

# examples of the word in use
print(syns[0].examples())

synonyms = []
antonyms = []


for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))  # set of synonyms for 'good'
print(set(antonyms))  # set of antonyms for 'good'


# Wu and Palmer method to compare the similarity of two words and their tenses
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))
