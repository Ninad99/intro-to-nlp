# Full tutorial - https://pythonprogramming.net/stop-words-nltk-tutorial/

'''
The idea of Natural Language Processing is to do some form of analysis, or processing, where the machine can understand, at least to some level, what the text means, says, or implies.

This is an obviously massive challenge, but there are steps to doing it that anyone can follow.
The main idea, however, is that computers simply do not, and will not, ever understand words directly. Humans don't either *shocker*.
In humans, memory is broken down into electrical signals in the brain, in the form of neural groups that fire in patterns.
There is a lot about the brain that remains unknown, but, the more we break down the human brain to the basic elements, we find out how basic the elements really are.

Well, it turns out computers store information in a very similar way!
We need a way to get as close to that as possible if we're going to mimic how humans read and understand text.
Generally, computers use numbers for everything, but we often see directly in programming where we use binary signals (True or False, which directly translate to 1 or 0, which originates directly from either the presence of an electrical signal (True, 1), or not (False, 0)).

To do this, we need a way to convert words to values, in numbers, or signal patterns.
The process of converting data to something a computer can understand is referred to as "pre-processing."
One of the major forms of pre-processing is going to be filtering out useless data.
In natural language processing, useless words (data), are referred to as stop words.

We would not want these words taking up space in our database, or taking up valuable processing time. As such, we call these words "stop words" because they are useless, and we wish to do nothing with them.
Another version of the term "stop words" can be more literal: Words we stop on.
'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# a set of stop words of the english language
stop_words = set(stopwords.words("english"))

print(stop_words)

sample_text_file = open("../sample.txt", "r")
text = sample_text_file.read()

word_tokens = word_tokenize(text)

filtered_sentence = []

for word in word_tokens:
    if word not in stop_words:
        filtered_sentence.append(word)

print("Filtered sentence:")
print(filtered_sentence)
