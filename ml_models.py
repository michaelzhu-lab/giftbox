import message_preprocessing

import tensorflow as tf

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import tqdm

import time
import pickle


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
import keras_metrics
import keras.backend as K
from keras.layers import Embedding, LSTM, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.naive_bayes import MultinomialNB



############################  NAVIE BAYES MODEL  ######################################
def ml_model(X, y, ngram_range, use_bag_of_words, mode, use_tf_idf, do_under_sampling, do_over_sampling, split_ratio,
           random_state):
    '''
    We only try naive bayes mode because our focus is not on using machine learning
    naive bayes model are used as a base model.
    
    example: 
    ngram_range (1,1), (2,2), (3,5) 
    use_bag_of_words = True
    mode = 'word', 'char'
    use_tf_idf = True
    do_under_sampling = True
    do_over_sampling = True
    
    '''
    # Get the n-gram or tf-idf matrix
    
    vectorizer_name = './models/tm_vectorizer_{}.pickle'.format(mode)

    import pickle
    
    if use_bag_of_words == True:
        vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer= mode)
        model_name = './models/nb_clf_tf_{}.pickle'.format(mode)
        n_gram_matrix = vectorizer.fit_transform(X['clean_msg'])
        
        vectorizer_name = './models/tm_vectorizer_{}.pickle'.format(mode)
        pickle.dump(vectorizer, open(vectorizer_name, 'wb'))
                
    elif use_tf_idf == True:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer= mode)
        model_name = './models/nb_clf_tfidf_{}.pickle'.format(mode)
        n_gram_matrix = vectorizer.fit_transform(X['clean_msg'])
        
        vectorizer_name = './models/tfidf_vectorizer_{}.pickle'.format(mode)
        pickle.dump(vectorizer, open(vectorizer_name, 'wb'))       
     
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, y, test_size = split_ratio, random_state=random_state)
    
    
    # print('X_train shape = {0}, y_train shape = {1}'.format(X_train.shape, y_train.shape))
    # print('X_test shape = {0}, y_test shape = {1}'.format(X_test.shape, y_test.shape))
    
    # Create a random over/under sampler that results in a 60/40 split
    if do_under_sampling:
        rs = RandomUnderSampler(sampling_strategy=40/60, random_state=random_state)
    elif do_over_sampling:
        rs = RandomOverSampler(sampling_strategy=40/60, random_state=random_state)
    
    if do_under_sampling or do_over_sampling:
        X_train, y_train = rs.fit_resample(X_train, y_train)
        y_train = y_train.reshape(-1,1)
        
        #print()
        #print('After resampling, X_train shape = {0}, y_train shape = {1}'.format(X_train.shape, y_train.shape))
        
        # pct = sum(y_train)/len(y_train)
        # pct = pct[0] * 100
        # print('New spam % in the training data = {}'.format(pct))
    
    #pct = sum(y_test.values)/len(y_test)
    #pct = pct[0] * 100
    #print()
    #print('Spam % in the test data = {}'.format(pct))
    #print()
    
    # Fit a Naive Bayes model
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    
    # save the model to disk
    pickle.dump(nb_clf, open(model_name, 'wb'))

    return nb_clf



############################  FUNCTION 2  ######################################
def show_model_stats(clf, X_train, X_test, y_train, y_test, name):
    # Predict on training and test data
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Create the confusion matrix

    # Training data
    print("Confusion Matrix: Training data")
    print(confusion_matrix(y_train, y_train_pred.reshape(-1,)))

    # Test data
    print("Confusion Matrix: Test data")
    print(confusion_matrix(y_test, y_test_pred.reshape(-1,)))

    # Classification report

    # Training data
    print("Classification report: Training data")
    print(classification_report(y_train, y_train_pred))

    # Test data
    print("Classification report: Test data")
    print(classification_report(y_test, y_test_pred))

    # Other metrics

    # Training data
    print("Metrics: Training data")
    print("Accuracy = {:.2f}".format(accuracy_score(y_train, y_train_pred)))
    print("Kappa = {:.2f}".format(cohen_kappa_score(y_train, y_train_pred)))
    print("F1 Score = {:.2f}".format(f1_score(y_train, y_train_pred)))
    print("Log Loss = {:.2f}".format(log_loss(y_train, y_train_pred)))

    # Test data
    print("\n")
    print("Metrics: Test data")
    print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_test_pred)))
    print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_test_pred)))
    print("F1 Score = {:.2f}".format(f1_score(y_test, y_test_pred)))
    print("Log Loss = {:.2f}".format(log_loss(y_test, y_test_pred)))

    # Class prediction error
    
    y_test = y_test.to_numpy()
    y_test = y_test.reshape(-1,)
    
    visualizer = ClassPredictionError(clf)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    g = visualizer.poof()

    fig = visualizer.ax.get_figure()
    fig.savefig('out/class-prediction-error-{}.png'.format(name), transparent=False)

    # ROC curve
    
    if clf.__class__.__name__ == 'SVC': # if SVM
        visualizer = ROCAUC(clf, micro=False, macro=False, per_class=False)
    else:
        visualizer = ROCAUC(clf)
    visualizer.fit(X_train, y_train) # fits the training data to the visualizer
    visualizer.score(X_test, y_test) # evaluate the model on test data
    g = visualizer.poof()

    fig = visualizer.ax.get_figure()
    fig.savefig('out/roc-curve-{}.png'.format(name), transparent=False)

    # Feature importance
    
    if hasattr(clf, 'feature_importances_'):
        features = X_train.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)

        fig = plt.figure(figsize=(10, 20))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')

        fig.savefig('out/feature-importances-{}.png'.format(name), transparent=False)
        
############################  BAGS + WORDS NEURAL NETWORK MODEL  ######################################