# parts of our preprocessing steps were adapted from https://towardsdatascience.com/text-summarization-from-scratch-using-encoder-decoder-network-with-attention-in-keras-5fa80d12710e

import contractions
import re
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import TextSummarizerModel
from collections import defaultdict
import gensim
import pickle

#Text Preprocessing
def clean_text(text, startend = False):
    clean_articles = []
    for article in text:
        expanded_words = [] 
        
        #expand contractions, remove punctuation, lowercase words
        for words in article.split():
            word = re.sub("\W+", "", words).lower()
            if word != "":
                expanded_words.append(word)
        article = " ".join(expanded_words)
        contractions.fix(article)
        #Add start and end tokens to articles
        if startend == True:
            article = "<s> " + article + " </s>"   
        
        clean_articles.append(article)
    return clean_articles

def embedding_matrix(x_tokenizer):
    #Get pretrained embeddings
    embeddings_wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    embeddings_wv.init_sims(replace=True)

    #Create our own embedding matrix with words in our articles   
    embeddings = np.zeros((len(x_tokenizer.word_index), 100))

    #if word is in pretrained embedding, take that embedding and add it to our matrix
    for word, index in x_tokenizer.word_index.items():
        try:
            embedding_vector = embeddings_wv[word]
            if embedding_vector is not None:
                embeddings[index] = embedding_vector
        except:
            pass
    return embeddings

# get word frequencies
def get_word_counts(text):
    d = defaultdict(int)
    for sent in text:
        for i in sent.split():
            d[i] += 1
    return d

# get count of rare words (words that appear less than cutoff)
def get_rare_words(d,cutoff):
    count = 0
    for k,v in d.items():
        if v < cutoff:
            count += 1
    return count

# save dataset 
def save_ds(og_x_text, og_y_text, x_text, y_text, x_seq, name = ''):
    with open(f'dataset{name}.pickle', 'wb') as f:
        pickle.dump((og_x_text, og_y_text, x_text, y_text, x_seq), f)


#preprocess and train data
def train(n):
    dataset_length = int(n*100/80)

    max_text_len = 500
    max_summary_len = 30
    units = 300

    dataset = load_dataset("xsum")
    ds = dataset['train'][:dataset_length]
    clean_doc = clean_text(ds['document'] )
    clean_sum = clean_text(ds['summary'],True)

    clean_doc = np.array(clean_doc)
    clean_sum = np.array(clean_sum) 

    #Perform 80/20 training test split
    x_train, x_test, y_train, y_test = train_test_split(clean_doc,clean_sum, test_size=0.2,shuffle=False)

    x_test_text, y_test_text = x_test, y_test
    og_x_train, og_x_test, og_y_train, og_y_test = train_test_split(ds['document'],ds['summary'], test_size=0.2,shuffle=False)

    # Get the number of common words
    x_d = get_word_counts(list(x_train))
    y_d = get_word_counts(list(y_train))

    x_num_words = len(x_d) - get_rare_words(x_d,100)
    y_num_words = len(y_d) - get_rare_words(y_d,10)

    # Get vocabulary sizes of x and y training sets
    x_size = x_num_words + 1
    y_size = y_num_words + 1

    #Create tokenizers for x_train and y_train
    x_tokenizer = Tokenizer(num_words = x_size, filters = '!"#$%&()*+,-.:;=?@[]^_`{|}~\t\n') 
    y_tokenizer = Tokenizer(num_words = y_size, filters = '!"#$%&()*+,-.:;=?@[]^_`{|}~\t\n')   

    x_tokenizer.fit_on_texts(list(x_train))
    y_tokenizer.fit_on_texts(list(y_train))

    # Convert training and test sets to sequences and pad them 
    x_train = pad_sequences(x_tokenizer.texts_to_sequences(x_train), maxlen=max_text_len, padding='post')
    x_test = pad_sequences(x_tokenizer.texts_to_sequences(x_test), maxlen=max_text_len, padding='post')

    y_train = pad_sequences(y_tokenizer.texts_to_sequences(y_train) , maxlen=max_summary_len, padding='post')
    y_test = pad_sequences(y_tokenizer.texts_to_sequences(y_test) , maxlen=max_summary_len, padding='post')


    embeddings = embedding_matrix(x_tokenizer)

    x_size, output_dim = embeddings.shape

    model = TextSummarizerModel(units, output_dim, x_size, y_size, x_tokenizer, y_tokenizer, max_text_len, max_summary_len, embeddings)
    model.forward()
    model.train(x_train,y_train,x_test,y_test)
    model.save_model(name = n)
    save_ds(og_x_test, og_y_test, x_test_text, y_test_text, x_test, name = n)

# train different models on 1000, 5000, 10000 , 50000 datapoints
for i in [1000, 5000, 10000 , 50000]:
    train(i)

