## import all necessary library
import datetime
import string
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import pickle

import unidecode
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud 

import warnings
warnings.simplefilter("ignore", UserWarning)
from matplotlib import pyplot as plt
# %matplotlib inline
# pd.options.mode.chained_assignment = None

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.externals import joblib

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC

import scipy
from scipy.sparse import hstack


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


############################  FUNCTION 1  ######################################
def text_load(file_name):
    '''
    load the text message file and save it into a data frame
    convert label columns into 0 (ham) ,1 (spam)
    return data frame
    '''
    path = '.\data'
    path_file = os.path.join(path, file_name)
    df = pd.read_csv(path_file, sep="\t", header=None)
    df.columns = ['is_spam', 'msg']
    df['is_spam'] = df['is_spam'].map({'ham': 0, 'spam': 1})
    return df

############################  FUNCTION 2  ######################################
def text_feature_engineer(df):
    '''
    input: raw message saved in the data frame
    output: dataframe with new feature
    1. CAPS
    2. ASCII
    3. PUNC
    4. LEN
    5. STOP
    6. NUM
    7. UNI
    8. URL
    '''
    feature_list = ['CAPS','ASCII','PUNC','LEN','STOP', 'NUM','UNI','URL']
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    for index, row in df.iterrows():
        df.at[index,feature_list[0]] = sum(1 for c in str(row['msg']) if c.isupper())
        df.at[index,feature_list[1]] = str.count(str(row['msg']), string.ascii_letters)
        df.at[index,feature_list[2]] = str.count(str(row['msg']), string.punctuation)
        df.at[index,feature_list[3]] = len(str(row['msg']))
        df.at[index,feature_list[4]] = sum([1 for w in row['msg'] if w.lower() in stopwords])
        df.at[index,feature_list[5]] = sum(w.isdigit() for w in row['msg'])
        df.at[index,feature_list[6]] = df.at[index,'LEN'] - df.at[index,'ASCII']
  
        regex = re.findall(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", 
                           str(row['msg']), re.IGNORECASE)
        
        # df.loc[df.URL != 1, 'URL'] = 0
        df[feature_list[7]] =0
        if regex != []:
            df.at[index,'URL'] = 1
            
    return df[feature_list]

############################  FUNCTION 3  ######################################
def text_cleaning(df):
    '''
    remove duplicated records
    clean all message
    '''
    # df.drop_duplicates(inplace=True)
    df['clean_msg'] = df['msg'].apply(clean_msg, args=(True, True, True, True, True, True))
    return df[['clean_msg']]

############################  FUNCTION 4  ######################################
def clean_msg(msg="", normalize_case=True, remove_punctuation=True, remove_non_unicode=True, remove_numbers=True,
              remove_stop_words=True, lemmatize_words=True):
    '''
    Input: a SMS text message
    Output: a cleaned SMS text message

    function
    1. case normalization: make lower case
    2. get rid of puctuation
    3. remove non-unicode
    4. remove numbers
    5. remove stop words
    6. lemmatization
    '''
        
    #stop_words = stopwords.words('english')
    stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
    lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()

    # 1. Case normalization: make lower case
    if normalize_case==True:
        msg = msg.lower()

    # 2. Get rid of punctuation
    if remove_punctuation==True:
        char_list = [char for char in msg if char not in string.punctuation]
        msg = ''.join(char_list)

    # 3. Remove non-unicode (ex: accents)
    if remove_non_unicode==True:
        msg = unidecode.unidecode(msg)

    # 4. Remove numbers
    if remove_numbers==True:
        char_list = [char for char in msg if not char.isdigit()]
        msg = ''.join(char_list)

    # 5. Remove stop words
    if remove_stop_words==True:
        words = msg.split(' ')
        words = [word for word in words if word not in stop_words]
        msg = ' '.join(words)

    # 6. Lemmatization (normalize across tenses, etc.)
    if lemmatize_words==True:
        words = msg.split(' ')
        words = [lemmatizer.lemmatize(word) for word in words]
        msg = ' '.join(words)

    # 7. Stemming (normalize across tenses, etc.)
    # Note that the main difference between stemming and lemmatization
    # is just that stemming operates without knowledge of the context.
    # The resultant "stems" may not be real words. However, through
    # lemmatization, the resultant "lemmas" are always valid words.
    # For example, the stem of the word "wolves" is "wolv", but the
    # lemma is "wolf".
    # if stem_words==True:
    #    words = msg.split(' ')
    #    words = [porter_stemmer.stem(word) for word in words]
    #    msg = ' '.join(words)

    return msg


############################  FUNCTION 5  #####################################
def text_tokenization(df_msg, maximum_len):
    '''
    vectorizing the message
    pad with 100 maxlen
    
    return vectorized message and tokenizer
    '''
    # Text tokenization: vectorizing text, turning each text into sequence of integers
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_msg)
    
    # convert to sequence of integers
    msg_seq = tokenizer.texts_to_sequences(df_msg)
    # padding with 100 maxlen
    msg_pad = pad_sequences(msg_seq, maxlen= maximum_len)
    
    pickle.dump(tokenizer, open('./models/tokenizer.pickle', 'wb'))
    
    return msg_pad, tokenizer

############################  FUNCTION 5  #####################################
def glove_embedding_matrix(tokenizer, embedding_dim):
    '''
    create a glove embedding matrix 
    '''    
    
    import os
    import pickle
    
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
    
    file_path = './models/glove_embed_matrix.pickle'
    if os.path.isfile(file_path):
        embedding_matrix = pickle.load(open(file_path, 'rb'))
    else:           
        embedding_index = {}
        with open('./glove_embedding/glove.6B.300d.txt',encoding='utf8') as f:
            for line in tqdm.tqdm(f, "Reading GloVe"):
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = vectors
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found will be 0s
                embedding_matrix[i] = embedding_vector

        pickle.dump(embedding_matrix, open('./models/glove_embed_matrix.pickle', 'wb'))
    
    return embedding_matrix