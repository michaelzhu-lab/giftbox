
import message_preprocessing
from message_preprocessing import text_feature_engineer
from message_preprocessing import text_load
from message_preprocessing import text_cleaning
from message_preprocessing import text_tokenization
from message_preprocessing import glove_embedding_matrix

import pickle
import numpy as np
import pandas as pd

import ml_models
import dl_models
from ml_models import ml_model
from dl_models import nn_model
from dl_models import cnn_model
from dl_models import rnn_model
from dl_models import rnn_cnn_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, precision_score, recall_score




# Load Data
# please save "SMSSpamCollection.txt" or other txt data in the "./data/"


# Train Model （hyper-parameters selected here are best results from grid-search）
## 1. ml_clf_word = Bag of words word-ngram + Navie Bayes 
## 2. ml_clf_char = Bag of words character-ngram + Navie Bayes
## 3. nn_clf = neural network + word embedding
## 4. nn_clf_glove = neural network + GloVe word embedding 
## 5. rnn_glove_clf = GloVe + RNN Model
## 6. cnn_glove_clf = GloVe + CNN
## 7. rnn_cnn_glove_clf = GloVe + RNN + CNN

MAXIMUM_LEN_MESSAGE = 100

def training_model(df, max_len = MAXIMUM_LEN_MESSAGE, embedding_dim = 300, normalize_case=True, remove_punctuation=True, 
                   remove_non_unicode=True, remove_numbers=True, remove_stop_words=True, lemmatize_words=True):
    '''
    training model
    
    default value:
    normalize_case=True,
    remove_punctuation=True,
    remove_non_unicode=True,
    remove_numbers=True,
    remove_stop_words=True,
    lemmatize_words=True
    
    all models    
    # 1. ml_clf_word = Bag of words word-ngram + Navie Bayes 
    # 2. ml_clf_char = Bag of words character-ngram + Navie Bayes
    # 3. nn_clf = neural network + word embedding
    # 4. nn_clf_glove = neural network + GloVe word embedding 
    # 5. rnn_glove_clf = GloVe + RNN Model
    # 6. cnn_glove_clf = GloVe + CNN
    # 7. rnn_cnn_glove_clf = GloVe + RNN + CNN      
    '''
    
    df.drop_duplicates(inplace=True)
    
    # Get the target variables
    df_Tgt = pd.get_dummies(df['is_spam'])
    
    # feature engineering is for ML and NN models
    df_FEng = text_feature_engineer(df)     # return new features (data frame)
    
    # clean 'msg' column
    df_msg = text_cleaning(df)     # return cleaned message (data frame)
      
    # tokenize cleaned message
    msg_pad, _tokenizer = text_tokenization(df_msg['clean_msg'], max_len)
    
    # add feature engineer
    msg_pad_FEng = np.concatenate((msg_pad, df_FEng.to_numpy()), axis=1)
    
    # glove_word_matrix
    glove_embed_matrix = pickle.load(open('./models/glove_embed_matrix.pickle', 'rb'))

    # split split ratio and random_state
    split_ratio = 0.25
    random_state = 7
    
    # 1. ml_clf_word = Bag of words word-ngram + Navie Bayes
    # ml_model(df_msg, df['is_spam'], (1,3), use_bag_of_words=False, 'word', use_tf_idf=True,
    #                       do_under_sampling=False, do_over_sampling=True, test_size = split_ratio, 
    #                       random_state = random_state)
    ml_clf_word = ml_model(df_msg, df['is_spam'], (1,3), False, 'word', True,False, True, split_ratio, random_state)
    
    # 2. ml_clf_char = Bag of words character-ngram + Navie Bayes
    # ml_model(df_msg, df['is_spam'], (1,3), use_bag_of_words=True, 'char', use_tf_idf=False, 
    #                       do_under_sampling=False, do_over_sampling=False, test_size = split_ratio, 
    #                       random_state = random_state)    
    ml_clf_char = ml_model(df_msg, df['is_spam'], (1,3), True, 'char', False, False, False, split_ratio, random_state)
    
    
    # 3. nn_clf = word embedding + neural network
    # nn_model(msg_pad, df_Tgt, test_size = split_ratio, random_state = random_state, Glove = False,
    #                  glove_embed_matrix, _tokenizer)
    nn_clf = nn_model(msg_pad_FEng, df_Tgt, split_ratio, random_state, glove_embed_matrix, _tokenizer, False)
    
    # 4. nn_clf_glove = word embedding (glove) + neural network
    # nn_model(msg_pad_FEng, df_Tgt, test_size = split_ratio, random_state = random_state, Glove = True, 
    #                        glove_embed_matrix, _tokenizer)
    nn_clf_glove = nn_model(msg_pad_FEng, df_Tgt, split_ratio, random_state, glove_embed_matrix, _tokenizer, True)
  
    # 5. rnn_glove_clf = GloVe + RNN Model
    # rnn_model(msg_pad, df_Tgt, _tokenizer, glove_embed_matrix, split_ratio, random_state, 
    #                    batch_size=64, epochs=20, lstm_units=128,sequence_length=max_len)
    rnn_glove_clf = rnn_model(msg_pad, df_Tgt,_tokenizer, glove_embed_matrix, split_ratio,random_state, 64, 20, 128,max_len)
    
    # 6. cnn_glove_clf = GloVe + CNN    
    # cnn_model(msg_pad, df_Tgt, _tokenizer, glove_embed_matrix, is_trainable_embedding=True, 
    #                          sequence_length=40, num_filters=64, filter_sizes=[5,7,9],dropout_rate=0.5,
    #                          conv_activation_fn='sigmoid', output_activation_fn='sigmoid', verbose==True, 
    #                          test_size = split_ratio, random_state = random_state,  batch_size=64, epochs=20                   
    cnn_glove_clf = cnn_model(msg_pad, df_Tgt, _tokenizer, glove_embed_matrix, True, max_len, 64,
                              [5,7,9],0.5, 'sigmoid','sigmoid', split_ratio, random_state, 64, 20, True)
    models_set ={
        "ml_clf_word": ml_clf_word,
        "ml_clf_char": ml_clf_char,
        "nn_clf": nn_clf,
        "nn_clf_glove": nn_clf_glove,
        "rnn_glove_clf": rnn_glove_clf,
        "cnn_glove_clf": cnn_glove_clf
    }
    
    return models_set

    

    # 7. rnn_cnn_glove_clf = GloVe + RNN + CNN
    ## due to time limitation, we haven't completed the debug of rnn_cnn_glove_clf; so we commentize the rnn_cnn_glove_clf 
    # rnn_cnn_glove_clf = rnn_cnn_model(msg_padd, df_Tgt, _tokenizer, glove_embedding_matrix, split_ratio, random_state)
    

def spam_message_predict(msg, maxlen = MAXIMUM_LEN_MESSAGE):      
    
    # retrain the model
    df = text_load("SMSSpamCollection.txt")
    saved_models = training_model(df, max_len = 100, embedding_dim = 300, normalize_case=True, remove_punctuation=True, 
                   remove_non_unicode=True, remove_numbers=True, remove_stop_words=True, lemmatize_words=True)
    
    # change msg into pandas data frame
    if msg == "":
        print("There is no message to predict")
        return ""
    if type(msg) == str:
        msg = pd.DataFrame(data = [msg], columns=['msg']) # save the mssage into dataframe
    elif type(msg)== list:
        msg = pd.DataFrame(data = msg, columns=['msg']) # save the mssage into dataframe
    else:
        print("The model can only predict str or [list of str]")
        return ""    
    
    tokenizer = pickle.load(open('./models/tokenizer.pickle', 'rb')) 
    
    # feature engineering is for ML and NN models
    df_FEng = text_feature_engineer(msg)     # return new features (data frame)
    # clean 'msg' column
    df_msg = text_cleaning(msg)     # return cleaned message (data frame)
    # tokenize cleaned message
    msg_pad = tokenizer.texts_to_sequences(df_msg['clean_msg'])
    msg_pad = pad_sequences(msg_pad, maxlen)
    # add feature engineer
    msg_pad_FEng = np.concatenate((msg_pad, df_FEng.to_numpy()), axis=1)
  
    char_vectorizer = pickle.load(open('./models/tm_vectorizer_char.pickle', 'rb'))
    word_vectorizer = pickle.load(open('./models/tfidf_vectorizer_word.pickle', 'rb'))
    char_gram_matrix = char_vectorizer.transform(df_msg['clean_msg'])
    word_gram_matrix = word_vectorizer.transform(df_msg['clean_msg'])    
    
    
    def spam_label(score_array):
        result = []
        for score in score_array:
            if score[0] >= score[1]:
                result.append("spam")
            else:
                result.append("ham")
        return result
    
    
    # make prediction
    our_prediction = {}
    our_prediction["ml_clf_word"] = ['ham' if i == 0 else 'spam' 
                                     for i in saved_models["ml_clf_word"].predict(word_gram_matrix).tolist()]
    our_prediction["ml_clf_char"] = ['ham' if i == 0 else 'spam'
                                     for i in saved_models["ml_clf_char"].predict(char_gram_matrix).tolist()]
    our_prediction["nn_clf"] = spam_label(saved_models["nn_clf"].predict(msg_pad_FEng))
    our_prediction["nn_clf_glove"] = spam_label(saved_models["nn_clf_glove"].predict(msg_pad_FEng))
    our_prediction["rnn_glove_clf"] = spam_label(saved_models["rnn_glove_clf"].predict(msg_pad))
    our_prediction["cnn_glove_clf"] = spam_label(saved_models["cnn_glove_clf"].predict(msg_pad))  

    # print the results
    for key, value in our_prediction.items():
        print("The model", key, "predict: ", value)



    
 