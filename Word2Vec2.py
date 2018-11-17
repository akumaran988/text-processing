import gzip
import gensim 
import logging
from gensim.parsing.preprocessing import remove_stopwords


f = open ('ETF.txt', 'r')
print('\n')
for i,line in enumerate (f):
    print(line)
    break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
 
    logging.info("reading file {0}...this may take a while".format(input_file))
    f = open(input_file, 'r')
    simple_pre_processed_text = []
    for i, line in enumerate(f):
 
        if (i % 10000 == 0):
            logging.info("read {0} reviews".format(i))
        # do some pre-processing and return list of words for each review
        # text
        line = remove_stopwords(line)
        simple_pre_processed_text.append(gensim.utils.simple_preprocess(line))
    return simple_pre_processed_text


simple_pre_processed_text = read_input('etf_self.txt')

print(simple_pre_processed_text)


model = gensim.models.Word2Vec(
        simple_pre_processed_text,
        size=100,
        window=4,
        min_count=5,
        workers=10)
model.train(simple_pre_processed_text, total_examples=len(simple_pre_processed_text), epochs=500, compute_loss=True)


print(model.wv.most_similar(positive='trade'))
print(model.wv.most_similar(positive='isin'))

print(model.get_latest_training_loss)


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)