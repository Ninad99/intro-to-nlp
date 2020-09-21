# Full tutorial - https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/

'''
One of the more powerful aspects of the NLTK module is the Part of Speech tagging that it can do for you.
This means labeling words in a sentence as nouns, adjectives, verbs...etc.
Even more impressive, it also labels by tense, and more. Here's a list of the tags, what they mean, and some examples:

POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''

from nltk.tag import pos_tag
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

sample_text_file = open("../sample.txt", "r")
text = sample_text_file.read()

pst = PunktSentenceTokenizer()

tokenized = pst.tokenize(text)


def process_content():
    try:
        for s in tokenized:
            words = word_tokenize(s)
            tagged = pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
