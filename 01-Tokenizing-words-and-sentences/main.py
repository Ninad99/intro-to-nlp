# Full tutorial - https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/

'''
Tokenizing - Splitting sentences and words from the body of text.

Token - Each "entity" that is a part of whatever was split up based on rules.
        For examples, each word is a token when a sentence is "tokenized" into words.
        Each sentence can also be a token, if you tokenized the sentences out of a paragraph.


'''

from nltk.tokenize import word_tokenize, sent_tokenize

sample_text_file = open("../sample.txt", "r")
text = sample_text_file.read()

print(word_tokenize(text))
print(sent_tokenize(text))
