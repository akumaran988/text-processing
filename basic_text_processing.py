import numpy as np
import re
from sklearn.svm import SVC


# Reading from a txt file
f = open('1.txt', 'r')
lines = f.readlines()
striped_lines = []
for line in lines:
    # Stripping the lines that are empty
    if line.strip():
        print('1' + line.lower())
        striped_lines.append(line.lower())


# Creating a dict with some values (important values)
local_dict = {'trade party': 0, 'isin': 1, 'value date': 2, 'trade date': 3}

# Creating a list of unprocessed words from the lines
unprocessed_words = []
for row, striped_line in enumerate(striped_lines):
    words = striped_line.split()
    for col, word in enumerate(words):
        print('['+str(row)+', '+str(col)+ ']    ' + word)
        unprocessed_words.append(word)

# Processing of unprocessed words for removal of varaiable data like alpha numerical and date etc..
processed_words = []
for i, unprocessed_word in enumerate(unprocessed_words):
    processed_word = re.sub(r':', '', unprocessed_word)
    processed_word = re.sub(r'[0-9][0-9]/[0-9][0-9]/[0-9][0-9]', '<unk>', processed_word)
    processed_word = re.sub(r'([a-z]{2})([0-9]{10})', '<unk>', processed_word)
    print(str(i)+' '+ processed_word)
    processed_words.append(processed_word)


# Creating a unique numerical value for a word in the dict
i = len(local_dict.items())
print("Dict count " + str(i))
for processed_word in processed_words:
    if not local_dict.keys().__contains__(processed_word):
        local_dict[processed_word] = i
        i = i+1

print('\nLocal dict\n')
for key in local_dict.keys():
    print(key + ' ' + str(local_dict[key]))


# Creating the one hot vector
train_input_vector = np.zeros((local_dict.items().__len__(), local_dict.items().__len__()),
                              float)

train_output_vector = np.zeros((local_dict.items().__len__(), ))
print("Train input vector shape: " + str(train_input_vector.shape))
print("Train output vector shape: " + str(train_output_vector.shape))

print('\nLocal dict\n')
for key in local_dict.keys():
    train_input_vector[local_dict[key], local_dict[key]] = 1
    if key == 'isin':
        train_output_vector[local_dict[key]] = 1

print("Train input vector: " + str(train_input_vector))
print("Train output vector: " + str(train_output_vector))

clf = SVC(kernel='poly', degree=20)
clf.fit(train_input_vector, train_output_vector)

print('\n\n')
print('print' + str(clf.predict(train_input_vector)))

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(50, input_dim=31, init='uniform', activation='relu'))
model.add(Dense(50, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model
model.fit(train_input_vector, train_output_vector, epochs=100, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(train_input_vector)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

# Testing

##
f = open('1.txt', 'r')
lines = f.readlines()

# converting all the values to lower case
processed_lines = []
for line in lines:
    processed_lines.append(line.lower())

processed_lines_numerical = []
for processed_line in processed_lines:
    for key in local_dict.keys():
        t_line = re.sub(key, str(local_dict[key]), processed_line)
        processed_line = t_line
    print('processed line:  ' + processed_line)
    processed_lines_numerical.append(processed_line)

##
