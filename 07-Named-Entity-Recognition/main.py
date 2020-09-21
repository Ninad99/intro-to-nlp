# Full tutorial - https://pythonprogramming.net/named-entity-recognition-nltk-tutorial/

'''
One of the most major forms of chunking in natural language processing is called "Named Entity Recognition."
The idea is to have the machine immediately be able to pull out "entities" like people, places, things, locations,
monetary figures, and more.

This can be a bit of a challenge, but NLTK is this built in for us. There are two major options with
NLTK's named entity recognition: either recognize all named entities, or recognize named entities as their respective type,
like people, places, locations, etc.

NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
'''

from nltk.tag import pos_tag
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import ne_chunk

sample_text_file = open("../sample.txt", "r")
text = sample_text_file.read()

pst = PunktSentenceTokenizer()

tokenized = pst.tokenize(text)


def process_content():
    try:
        for s in tokenized:
            words = word_tokenize(text)
            tagged = pos_tag(words)
            named_entities = ne_chunk(tagged)
            named_entities.draw()
            # print(named_entities)

    except Exception as e:
        print(str(e))


process_content()
