import nltk
import collections 
import string

"""
 BMIG5003 Monday, October 25 and Wednesday, October 27
 Example on working with corpora and generating bigrams. 
 Based in part on https://www.nltk.org/book/ch02.html
 Requires NLTK
"""

# Download the project gutenberg corpus
nltk.download('gutenberg')


# simple example sentence and very simple tokenization/bigrams
foxs = "the quick brown fox is a fast brown fox"
foxbg = list(nltk.bigrams(foxs.split()))

# print the bigrams
print(f"Fox bigrams: {foxbg}")
# print the bigram count
print(f"Fox bigram Counter: {collections.Counter(foxbg)}")

# -----------------



# get the list of "words" in Alice in Wonderland -- this includes many non-word tokens, as we saw!
cawords = nltk.corpus.gutenberg.words('carroll-alice.txt')

# get the bigrams
# note that nltk.bigrams returns a generator. We store the result a list so we can reuse it
cabgl = list(nltk.bigrams(cawords))


print(f"\n\nNum words in lewis carroll's alice: {len(list(cawords))}")
print(f"Num bigrams in order in lewis carroll's alice: {len(list(cabgl))}")
print(f"Num unique bigrams in lewis carroll's alice: {len(set(cabgl))}")

"""
 generate bigrams for the words in 'Alice', and count them using
https://docs.python.org/3/library/collections.html#collections.Counter
"""
c = collections.Counter(list(cabgl))

csorted = sorted(c.items(), key = lambda x : x[1], reverse = True)
print(f"The 20 most common bigrams: {csorted[:20]}")



"""
Try with stopwords 
"""
from nltk.corpus import stopwords
stops = stopwords.words('english') + list(string.punctuation)
cawords2 = [w for w in cawords if not w in stops]

cabgl2 = list(nltk.bigrams(cawords2))

print("\n\nAfter stopword removal")
print(f"Num words in lewis carroll's alice: {len(list(cawords2))}")
print(f"Num bigrams in order in lewis carroll's alice: {len(list(cabgl2))}")
print(f"Num unique bigrams in lewis carroll's alice: {len(set(cabgl2))}")

c2 = collections.Counter(list(cabgl2))

csorted2 = sorted(c2.items(), key = lambda x : x[1], reverse = True)
print(f"\n\nThe 20 most common bigrams after removing stopwords: {csorted2[:20]}")

#-------------------



"""
Several of the most common "words" in these bigrams are actually 
still just sequence of punctuation. 
Let's try to get rid of those using the function we developed in class on Wednesday, October 27

That (quick and dirty) function drops punctuation from strings. 

Examples:
>>> drop_nasty_punct("Alice!")
'Alice'

>>> drop_nasty_punct("$#!")
''
"""
def drop_nasty_punct(s):
    return ''.join([w for w in s if not w in list(string.punctuation)])


""" Use dnp to get rid of the weird punctuation-only tokens. 
    This list comprehension relies on the fact that the empty string is considered False in boolean expression in Python!
"""
cawords3 = [w for w in cawords if (not w in stops) and drop_nasty_punct(w) ]
cabgl3 = list(nltk.bigrams(cawords3))
c3 = collections.Counter(list(cabgl3))
csorted3 = sorted(c3.items(), key = lambda x : x[1], reverse = True)

print(f"The 20 most common bigrams after removing nasty punctuation tokens: {csorted3[:20]}")




