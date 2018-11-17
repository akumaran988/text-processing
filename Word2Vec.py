from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 

warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec 


sample = open("2.txt", "r") 
s = sample.readlines() 

# Replaces escape character with space 
#f = s.replace("\n", " ") 
print(s)

data1 = [] 

# iterate through each sentence in the file 
for i in s: 
	temp1 = [] 
	
	# tokenize the sentence into words 
	for j in i.split(): 
		temp1.append(j.lower()) 

	data1.append(temp1) 

from gensim.corpora import Dictionary

corpus = ["máma mele maso".split(), "ema má máma".split()]
dct = Dictionary(corpus)
print(len(dct))
corpus = open("customwords.txt", 'r').readlines()
corpus = [[i.replace('\n','')] for i in corpus]
dct = Dictionary(corpus)
print(len(dct))
dct.add_documents([["this", "is", "sparta"], ["just", "joking"]])
print(len(dct))
# Create CBOW model 
model1 = gensim.models.Word2Vec(data1, min_count = 1, size = 100, window = 2) 

print(v)
# Print results 
print("Cosine similarity between 'Trade' " +
			"and 'Confirmation' - CBOW : ", 
	model1.similarity('Trade', 'Confirmation')) 
	
print("Cosine similarity between 'Trade' " +
				"and '' - CBOW : ", 
	model1.similarity('Trade', 'Confirmation')) 

# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data1, min_count = 1, size = 100, window = 5, sg = 1) 
print(model2.vocabulary.value)

