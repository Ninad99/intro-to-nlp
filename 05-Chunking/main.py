# Full tutorial - https://pythonprogramming.net/chunking-nltk-tutorial/
# Regex tutorial - https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/

'''
Once we know the parts of speech, we can do what is called chunking, and group words into hopefully meaningful chunks.
One of the main goals of chunking is to group into what are known as "noun phrases."

These are phrases of one or more words that contain a noun, maybe some descriptive words, maybe a verb,
and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.

In order to chunk, we combine the part of speech tags with regular expressions.
Mainly from regular expressions, we are going to utilize the following:

+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line
'''

from nltk.tag import pos_tag
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import RegexpParser

sample_text_file = open("../sample.txt", "r")
text = sample_text_file.read()

pst = PunktSentenceTokenizer()

tokenized = pst.tokenize(text)


def process_content():
    try:
        for s in tokenized:
            words = word_tokenize(s)
            tagged = pos_tag(words)
            chunkGram = r"""Chunk: {<VB.?>*<NNP>+<NN>?}"""
            # look for any verb, atleast one proper noun and zero or one noun
            chunkParser = RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            # print(chunked)

    except Exception as e:
        print(str(e))


process_content()
