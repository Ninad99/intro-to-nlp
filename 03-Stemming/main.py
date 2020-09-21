# Full tutorial - https://pythonprogramming.net/stemming-nltk-tutorial/

'''
The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning,
other than when tense is involved.

The reason why we stem is to shorten the lookup, and normalize sentences.

Consider:

I was taking a ride in the car.
I was riding in the car.

This sentence means the same thing. in the car is the same. I was is the same.
the ing denotes a clear past-tense in both cases, so is it truly necessary to differentiate between ride and riding,
in the case of just trying to figure out the meaning of what this past-tense activity was?

No, not really.

This is just one minor example, but imagine every word in the English language, every possible tense and
affix you can put on a word. Having individual dictionary entries per versionwould be highly redundant and inefficient,
especially since, once we convert to numbers, the "value" is going to be identical.

One of the most popular stemming algorithms is the Porter stemmer, which has been around since 1979.
'''

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["call", "caller", "calling", "called"]

for w in example_words:
    print(ps.stem(w))

example_sentence = "An unknown person called me yesterday. He was the caller but he acted like I was the one calling him! What a weird feeling to be called like that!"

words = word_tokenize(example_sentence)

for w in words:
    print(ps.stem(w))
