# Full tutorial - https://pythonprogramming.net/chinking-nltk-tutorial/

'''
Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk.
The chunk that you remove from your chunk is your chink.

The code is very similar, you just denote the chink, after the chunk, with }{ instead of the chunk's {}.
'''

from nltk import RegexpParser
from nltk.tag import pos_tag
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
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            chunkParser = RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            # print(chunked)

    except Exception as e:
        print(str(e))


process_content()
