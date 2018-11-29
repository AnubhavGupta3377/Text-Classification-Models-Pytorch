# utils.py

from nltk import word_tokenize
from tqdm import tqdm
from gensim.models import KeyedVectors
import numpy as np
import math

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown = '<unk>'
        self.add_word(self.unknown)
        
    def add_word(self, word):
        ''' Add a word to the vocabulary
        Inputs:
            word (string) : Word to be added to vocabulary
        '''
        
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
    
    def construct(self, words):
        ''' Construct the vocabulary
        Inputs:
            words (list[string]) : List of words defining the vocabulary
        '''
        
        for word in words:
            self.add_word(word)
        print ("Constructed vocabulary of size: {}".format(len(self.word_to_index)))
        
    def encode(self, word):
        '''
        Given a word, get corresponding index in vocabulary.
        '''
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index(word)
    
    def decode(self, index):
        '''
        Given a word index, get corresponding word from vocabulary.
        '''
        return self.index_to_word(index)
    
    def __len__(self):
        return len(self.word_to_index)

def parse_label(label):
    '''
    Get the actual labels from label string
    Input:
        label (string) : labels of the form '__label__2'
    Returns:
        label (int) : integer value corresponding to label string
    '''
    return int(label.strip()[-1])

def get_data(filename):
    ''' Loads the data from file
    Inputs:
        filename (String): absolute path to the datafile
    Returns:
        X (list[string]): list of contents of documents
        y (list[integer]): labels of documents
        vocab : Vocab object corresponding to words in X
    '''
    
    print ('Reading data from {}'.format(filename))
    with open(filename, 'r') as datafile:     
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: parse_label(x[0]), data))
    vocab = Vocab()    
    words = set(' '.join(data_text).lower().split())
    vocab.construct(list(words))
    return data_text, data_label, vocab

def get_word_embeddings(w2vfile, word_to_index, embedsize=300):
    '''
    For each word in our vocabulary, get the word2vec encoding of the word
    Inputs:
        w2vfile (string) : Path to the file containing (pre-trained) word embeddings
        embedsize (int) : Length of each word vector
    Returns:
        word_embeddings : Dictionary mapping each word to corresponding embedding
    '''

    word_embeddings = {}
    if w2vfile.endswith('.txt'):
        f = open(w2vfile)
        for line in tqdm(f):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if word in word_to_index:
                word_embeddings[word] = coefs
        f.close()
    elif w2vfile.endswith('.bin'):
        word2vec = KeyedVectors.load_word2vec_format(w2vfile, binary=True, limit=1000000)
        for word in tqdm(word_to_index):
            try:
                word_embeddings[word] = word2vec[word.lower()]
            except KeyError:
                pass
    else:
        print ('Can\'t load word embeddings.')
        exit(-1)

    print('Found {0}/{1} word vectors.'.format(len(word_embeddings), len(word_to_index)))
    if len(word_to_index) > len(word_embeddings):
        print('Initializing remaining {} word vectors with zeros.'.format(len(word_to_index) - len(word_embeddings)))

    for word in word_to_index:
        if word not in word_embeddings:
            word_embeddings[word] = np.zeros((embedsize,))
    return word_embeddings

def encode_text(text, word_embeddings, max_sen_len):
    '''
    Encode a sequence of words into corresponding vector representation
    Input:
        text (string) : text (space separated words, etc..)
        word_embeddings (dict) : dictionary mapping from words to their representation
        max_sen_len (int) : maximum sentence length (in words)
    Returns:
        X (np.matrix) : matrix of shape (max_sen_len, embedding_size) after zero padding
    '''
    
    default_embed = np.zeros(300)
    words = word_tokenize(text)[:max_sen_len]
    embeds = [word_embeddings.get(x, default_embed) for x in words]
    embeds += [default_embed] * (max_sen_len - len(embeds))
    return np.array(embeds, dtype=np.float32)

def data_iterator(train_x, train_y, batch_size = 256):
    '''
    Generate batches of training data for training (for single epoch)
    Inputs:
        train_df (pd.DataFrame) : complete training data
        batch_size (int) : Size of each batch
    Returns:
        text_arr (np.matrix) : Matrix of shape (batch_size,embed_size)
        lebel_arr (np.array) : Labels of this batch. Array of shape (batch_size,)
    '''
    n_batches = math.ceil(len(train_x) / batch_size)
    for idx in range(n_batches):
        x = train_x[idx *batch_size:(idx+1) * batch_size]
        y = train_y[idx *batch_size:(idx+1) * batch_size]
        yield x, y