import joblib
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss, precision_score, recall_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Embedding, LSTM, Dropout, Dense
import keras_metrics

from keras.utils import to_categorical

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

############################  All models  ######################################


############################  nn_clf  #####################################
def nn_model(X, y, split_ratio, random_state, glove_embedding_matrix, tokenizer, Glove ):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split_ratio, random_state=random_state)
    
    if Glove == True:
        model_name = './models/nn_clf_Glove_embedding.pickle'
    else:
        model_name = './models/nn_clf_word_embedding.pickle'
    
    model = Sequential()
    
    if Glove == True:
        model.add(Embedding(len(tokenizer.word_index)+1, glove_embedding_matrix.shape[1], 
                            weights=[glove_embedding_matrix], trainable=True, input_length=108))
    else:
        matrix_size = glove_embedding_matrix.shape
        random_embedding_matrix = np.random.random(matrix_size)
        model.add(Embedding(len(tokenizer.word_index)+1, random_embedding_matrix.shape[1], 
                            weights=[random_embedding_matrix], trainable=True,input_length=108))
     
    model.add(Flatten())
    # Add the second hidden layer
    model.add(Dense(50, activation='relu'))

    # Add the second layer
    model.add(Dense(8, activation='relu'))

    # Add the output layer
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer = "adam",loss='categorical_crossentropy',
                  metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=64, epochs=20, verbose=1)
    
    # save the model to disk
    pickle.dump(model, open(model_name, 'wb'))
        
    return model


############################  cnn_clf  #####################################

def cnn_model(X, y, tokenizer, embedding_matrix, is_trainable_embedding, sequence_length, num_filters, filter_sizes,
              dropout_rate, conv_activation_fn, output_activation_fn, split_ratio, random_state, batch_size, epochs, verbose):
    # Split into training and test sets
    
    model_name = './models/cnn_clf_Glove_embedding.sav'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
             
    # Construct the CNN layers and build the model (using the functional API)
    EMBEDDING_SIZE = embedding_matrix.shape[1]
    
    inputs = Input(shape=(sequence_length,))
    embedding = Embedding(len(tokenizer.word_index)+1,
                          EMBEDDING_SIZE,
                          weights=[embedding_matrix],
                          trainable=is_trainable_embedding,
                          input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, EMBEDDING_SIZE, 1))(embedding)
    
    # Note that our filters are the length of the embedding vectors so that full words appear in each filter
    
    conv2d_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_SIZE), activation=conv_activation_fn)(reshape)
    conv2d_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv2d_0)
    
    conv2d_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_SIZE), activation=conv_activation_fn)(reshape)
    conv2d_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv2d_1)
    
    conv2d_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_SIZE), activation=conv_activation_fn)(reshape)
    conv2d_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv2d_2)
    
    # Add the convolutional layers together
    
    conv2d_all = Concatenate(axis=1)([conv2d_0, conv2d_1, conv2d_2])
    
    # Flatten, dropout, and a final dense layer
    
    conv2d_all = Flatten()(conv2d_all)
    conv2d_all = Dropout(dropout_rate)(conv2d_all)
    conv2d_all = Dense(units=2, activation=output_activation_fn)(conv2d_all)
    
    # Put it all together and compile the model
    
    cnn_clf = Model(inputs=inputs, outputs=conv2d_all)
    cnn_clf.compile(optimizer='adam', loss='binary_crossentropy', 
                    metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    
    # Fit the model and get the best one (as per validation accuracy)
    
    model_checkpoint = ModelCheckpoint('./cnn_spam_classifier.hdf5', monitor='val_accuracy', 
                                       mode='max', save_best_only=True, verbose=verbose)
    
    print(str(X_train.shape))
    print(str(y_train.shape))
    print(str(X_test.shape))
    print(str(y_test.shape))
    
    cnn_clf.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, 
                epochs=epochs, callbacks=[model_checkpoint], verbose=verbose)
    
    #best_cnn_clf = load_model('./cnn_spam_classifier.hdf5', 
    #                          custom_objects={'binary_precision':keras_metrics.binary_precision(), 
    #                                          'binary_recall':keras_metrics.binary_recall()})
    
    # save the model to disk
    pickle.dump(cnn_clf, open(model_name, 'wb'))
    
    return cnn_clf


############################  rnn_clf  #####################################
def rnn_model(X, y, tokenizer, embedding_matrix, split_ratio, random_state, batch_size, epochs,lstm_units,sequence_length):
     
    model_name = './models/rnn_clf_glove_embedding.sav'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
        
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,embedding_matrix.shape[1],weights=[embedding_matrix],
              trainable=True,input_length=sequence_length))    

    model.add(LSTM(lstm_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    # compile as rmsprop optimizer
    # aswell as with recall metric
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])
        #model.summary()
    
    
    model_checkpoint = ModelCheckpoint('./rnn_spam_classifier.hdf5', monitor='val_accuracy', 
                                       mode='max', save_best_only=True, verbose=True)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, 
                epochs=epochs, callbacks=[model_checkpoint], verbose=True)      
        
    #best_rnn_clf = load_model('./rnn_spam_classifier.hdf5', 
    #                          custom_objects={'binary_precision':keras_metrics.binary_precision(), 
    #                                          'binary_recall':keras_metrics.binary_recall()})
   
    pickle.dump(model, open(model_name, 'wb'))
    
    return model


############################  rnn_clf  #####################################
def rnn_cnn_model(X, y, tokenizer, embedding_matrix, split_ratio, random_state, batch_size, epochs, sequence_length):
    
    model_name = './models/rnn_cnn_clf_glove_embedding.sav'
    
    embedding_dim = embedding_matrix.shape[1]
    inp = Input(shape=(sequence_length, ))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
 
    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(2, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model_checkpoint = ModelCheckpoint('./rnn_cnn_spam_classifier.hdf5', monitor='val_accuracy', 
                                       mode='max', save_best_only=True, verbose=verbose)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, 
                epochs=epochs, callbacks=[model_checkpoint], verbose=True)      
        
    #best_rnn_cnn_clf = load_model('./rnn_cnn_spam_classifier.hdf5', 
    #                          custom_objects={'binary_precision':keras_metrics.binary_precision(), 
    #                                          'binary_recall':keras_metrics.binary_recall()})    
    
    pickle.dump(model, open(model_name, 'wb'))
     
    return model

