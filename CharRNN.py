import pandas as pd
import numpy 
from keras.datasets import imdb 
from keras.models import Sequential 
from keras.layers import Dense, Flatten
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence

# fix random seed for reproducibility 
numpy.random.seed(7) 

# load the dataset but only keep the top n words, zero the rest 
top_words = 5000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

import keras
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in X_train[0] ))


from numpy import array
from keras.preprocessing.text import one_hot
docs = ['Gut gemacht',
		'Gute arbeit',
		'Super idee',
		'Perfekt erledigt',
		'exzellent',
		'naja',
		'Schwache arbeit.',
		'Nicht gut',
		'Miese arbeit.',
		'Hätte es besser machen können.']
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

# truncate and pad the review sequences 
max_review_length = 500 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

pd.DataFrame(X_train).head()

# create the model 
embedding_vector_length = 32 
model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length)) 
#model.add(LSTM(100)) 
model.add(Flatten()) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))

bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
for review in [good,bad]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
    print("%s . Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))