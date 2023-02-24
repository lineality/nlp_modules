"""
We Have: a raw string sentence input
We Want: a bag of words (BOW) model
We Need: 

Context: 
 Following NLPIA by Hobbson Lane, Manning publisher
 Chapter 2

Review & Refactor:

Requires:
- numpy
- pandas

Recommended:
- python venv env
- python v 3.11.1

"""


# import libraries
import numpy as np
import pandas as pd

#########
# v2.1.1
#########
print( "\nPart 1" )
sentence = """Thomas Jefferson began building Montecello at the age of 26."""

output = sentence.split()

print( output )


#########
# v2.1.2
#########
print( "\nPart2" )
# import numpy as np
token_sequence = str.split( sentence )
vocab = sorted( set( token_sequence ) )

output = ' ,'.join(vocab)

print( output )

num_tokens = len( token_sequence )

vocab_size = len( vocab )

onehot_vectors = np.zeros( (num_tokens, vocab_size), int)

# use enumerate to set one=hot, one=yes, values
for this_index, this_word in enumerate( token_sequence ): 
    onehot_vectors[this_index, vocab.index( this_word )] = 1

output = ' '.join( vocab )

print( output )

print( onehot_vectors )


######
# 2.2
######
print( "\nPart 3" )
"""
using pandas
"""

# use existing numpy array to make a pandas dataframe
df = pd.DataFrame( onehot_vectors, columns=vocab )

print( df )


###########################
# Side Fun 1: player piano
###########################
print("\nSide Fun: see Bag of Words as a player piano scroll")
pp_df = df.copy()

pp_df[ pp_df == 0 ] = ''

print( pp_df )

#########
# Tokens
#########
print( "\nPart 4: Simple Tokens" )

# make a bow set (no duplicates)
sentence_bow = {}

for token in sentence.split():
    sentence_bow[token] = 1

output = sorted( sentence_bow.items() )

print( output )

#####################################
# Better tokens with pandas (series)
#####################################
# note ->  .T = transverse
df = pd.DataFrame( pd.Series( dict( [(token, 1) for token in sentence.split() ])), columns=['sent']).T

print( df )

print("note: now a row is a sentence or a document")

#############################
# "corpus" and B.O.W. Vectors (dataframe)
#############################
print("\nCorpus and vectors")

# note: would be nice to read in a .txt file here...

sentences = """Thomas Jeffersn began building Montecello at the age of 26.
Construction was done mostly by local masons and carpenters.
He moved int the South Pavilion in 1770.
Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."""

# set of multiple documents called a 'corpus'
corpus = {}

# enumerate and splint on new-line
for this_index, this_sentence in enumerate( sentences.split('\n') ):
    corpus[ 'this_sentence{}'.format(this_index)] = dict( (tok, 1) for tok in this_sentence.split() )

# handle missing values cleanly as integer zeros
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

print( df )

# page 42
