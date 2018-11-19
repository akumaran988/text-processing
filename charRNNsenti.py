import numpy as np
from keras.layers import Dense, Flatten, LSTM
from keras.models import Sequential 
from keras.preprocessing import sequence
from keras.utils import to_categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
common_words  = open('words.txt').readlines()

Common_X_train = []

for common_word in common_words:
    sample = []
    for i, chara in enumerate(common_word):
        sample.append(np.array([ord(chara)]))
    Common_X_train.append(np.array(sample))

Common_X_train = np.array(Common_X_train)
print(Common_X_train)

Common_X_test = Common_X_train

Common_y_train = np.zeros((Common_X_train.shape[0]))

Common_y_train = to_categorical(Common_y_train, 5)
Common_y_test = Common_y_train


#%%
text = "trade date\ncurrency\nprice\nvalue date"

words = text.split('\n')

X_train = []

for word in words:
    sample = []
    for i, chara in enumerate(word):
        sample.append(np.array([ord(chara)]))
    X_train.append(np.array(sample))

X_train = np.array(X_train)
print(X_train)

y_train = np.array([1, 2,  3, 4])

y_train = to_categorical(y_train, 5)


X_train = np.concatenate((Common_X_train, X_train), axis=0)
y_train = np.concatenate((Common_y_train, y_train), axis=0)

X_test = X_train
y_test = y_train
max_review_length = 10 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


print(X_train.shape)
print(y_train.shape)

 

# create the model 
model = Sequential()
model.add(LSTM(300, input_shape=(10,1), return_sequences=True))
model.add(LSTM(500))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=200)

# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))

text = "trade date\ncurrency\nfor\nqwert\nprice"

words = text.split('\n')

X_test = []

for word in words:
    sample = []
    for i, chara in enumerate(word):
        sample.append(np.array([ord(chara)]))
    X_test.append(np.array(sample))

X_test = np.array(X_test)
print(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


Xtest_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('result')
print(model.predict(Xtest_padded))

from keras.models import load_model

model.save('my_model.h5')

#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')