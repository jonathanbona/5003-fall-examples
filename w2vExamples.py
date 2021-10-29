import gensim
from gensim.models import KeyedVectors

"""
Loading and using pre-trained word2vec vectors from google

"""

# Google pre-trained vectors, available here: https://code.google.com/archive/p/word2vec/
# To use you will need to download these and put them in a place python can find them
googlevecs = './data/GoogleNews-vectors-negative300.bin.gz'
word_vectors = KeyedVectors.load_word2vec_format(googlevecs, binary=True, limit=200000)


""" Example interactions showing use of these pre-trained vectors
>>> word_vectors.similar_by_word('king')
[('kings', 0.7138045430183411), ('queen', 0.6510956883430481), ('monarch', 0.6413194537162781), ('crown_prince', 0.6204220056533813), ('prince', 0.6159993410110474), ('sultan', 0.5864824056625366), ('ruler', 0.5797567367553711), ('princes', 0.5646552443504333), ('throne', 0.5422105193138123), ('royal', 0.5239794254302979)]

>>> word_vectors['codeine']
array([ 0.00141144, -0.33203125, -0.15039062, -0.08496094,  0.36523438,
        ...
        0.17480469,  0.19726562,  0.20605469,  0.28710938, -0.13476562],
      dtype=float32)

>>> word_vectors.similar_by_word('codeine')
[('cough_syrup', 0.5610266327857971), ('morphine', 0.5592806935310364), ('hydrocodone', 0.5402147769927979), ('painkillers', 0.5401848554611206), ('Valium', 0.5272670388221741), ('prescription_painkiller', 0.5253542065620422), ('oxycodone', 0.519694983959198), ('Vicodin', 0.5153637528419495), ('ketamine', 0.514746904373169), ('Hydrocodone', 0.5111684203147888)]
"""



import pandas as pd
from nltk import casual_tokenize
import gensim

# EMAIL jpbona@uams.edu for this file
inf = './1557tweets.csv'
entries = pd.read_csv(inf).text


# a list of lists of tokens (strings). each inner list is the tokens for a sentence
toksents = [nltk.casual_tokenize(s) for s in entries]

# Train a lousy w2v model for comparison
model1 = gensim.models.Word2Vec(toksents,
                                vector_size=150,    # our vectors will have 150 features 
                                window=5,    # how far to look from each word when training
                                min_count=2, # "Ignores all words with total frequency lower than this."
                                workers=10,  # threads
                                epochs=1)      # how many iterations. 1 is going to be too few


# Train a better model
model150 = gensim.models.Word2Vec(toksents,
                                vector_size=150,
                                window=5,   
                                min_count=2,
                                workers=10,
                                epochs=150)


"""Example interactions
>>> model1.wv.most_similar('codeine')
[('to', 0.42450255155563354), ('the', 0.39510008692741394), ('I', 0.387450248003006), ('opium', 0.38621705770492554), ('.', 0.35779234766960144), (',', 0.3525272309780121), ('you', 0.3269294202327728), ('tramadol', 0.3259072005748749), ('s', 0.3252558410167694), ('-', 0.30906328558921814)]

>>> model150.wv.most_similar('codeine')
[('Codeine', 0.5973665118217468), ('sprite', 0.4339492917060852), ('💥', 0.43129414319992065), ('Tylenol', 0.40488800406455994), ('acetaminophen', 0.37919875979423523), ('hood', 0.36339399218559265), ('remember', 0.3601214587688446), ("syrup's", 0.35976094007492065), ( 'lean', 0.35732021927833557), ('syrup', 0.35650894045829773)]

>>> model150.wv.most_similar('💥')
[('😇', 0.6344366073608398), ('red', 0.6159301996231079), ('Codeine', 0.5990944504737854), ('yeah', 0.5916265845298767), ('Cody', 0.5582106709480286), ('🎶', 0.5397436022758484), ('cups', 0.5374603271484375), ('fuckin', 0.5352113842964172), ('moving', 0.5177707076072693), ('Caffeine', 0.5150325298309326)]
"""
