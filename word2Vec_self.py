import gensim 
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np

from gensim.corpora import Dictionary


f = open ('etf_self.txt', 'r')
print('\n')
for i,line in enumerate (f):
    print(line)
    break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
 
    f = open(input_file, 'r')
    simple_pre_processed_text = []
    for i, line in enumerate(f):
        line = remove_stopwords(line)
        simple_pre_processed_text.append(gensim.utils.simple_preprocess(line))
    return simple_pre_processed_text


simple_pre_processed_text = read_input('etf_self.txt')

print(simple_pre_processed_text)

def generate_word2vec_train_samples(input_list, window_size):
    clean_list = []
    train_vec = []
    for i in input_list:
        if len(i)>0:
            clean_list.append(i)
    for sentence_index in range(len(input_list)):
        for word_index in range(len(input_list[sentence_index])):
            if(word_index+1 > window_size and word_index < len(input_list[sentence_index])-window_size):
                for i in np.arange(-1 * window_size, window_size+1, 1):
                    #print('sentence_length ' + str(len(input_list[sentence_index]))+' '+'word_index = '+ str(word_index)+ ' ' +str(word_index + i))
                    train_vec.append([input_list[sentence_index][word_index], input_list[sentence_index][word_index+i]])
    return train_vec



train_samples = generate_word2vec_train_samples(simple_pre_processed_text, 2)
print(train_samples)
local_dict = {'trade':0, 'etf':1, 'bond':2, 'isin':3, 'isincode':4, 'code':5, 'identification':6,
            'notional':7, 'nominal': 8, 'quantity': 9, 'sell':10, 'sold':11, 'amount':12, 'price':13, 'cost':14}


train_input_vector = np.zeros((len(train_samples), local_dict.items().__len__()),
                              float)

train_output_vector = np.zeros((len(train_samples), local_dict.items().__len__()),
                              float)
print("Train input vector shape: " + str(train_input_vector.shape))
print("Train output vector shape: " + str(train_output_vector.shape))

print('\nLocal dict\n')
for i, sample in enumerate(train_samples):
    for key in local_dict.keys():
        if (sample[0] == key):
            train_input_vector[i, local_dict[key]] = 1
        if (sample[1] == key):
            train_output_vector[i, local_dict[key]] = 1
        

print("Train input vector: " + str(train_input_vector))
print("Train output vector: " + str(train_output_vector))


# Converting the output label for multi class
inputs = train_output_vector
print(inputs.shape[0])
