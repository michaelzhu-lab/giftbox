# giftbox
Data Science journey 
The first project is a spam detection using deep learing and Naive bayes method with @MarkSmith, @JinWang, @Suky,@Janki, @TobiasBoadway, and @Sekar

We implements four kinds of methods for this famous spam detection problem and compared the results
        1. Naive Bayes
        2. Neural Network
        3. Recurennt Neural Network (RNN)
        4. Convolutional Neural Netwrok (CNN)
        
For the pre-trained emeddding glove, we used the GloVe: Global Vectors for Word Representation with 300 dimension , which you can download it here https://nlp.stanford.edu/projects/glove/
# How was the model saved
 The spam classification model was saved in four different python files as follows:
1.  	message_preprocessing.py: It covers data cleaning, tokenization, creation of word embedding.
2.  	ml_models.py: It includes all Naive Bayes based machine learning models
3.  	dl_models.py: It includes all deep learning modes: artificial neural network model, convolutional network models and recurrent network models
4.  	prediction.py: It includes the model training and model prediction.
# How to make a prediction
1.  	First need to import prediction
2.  	Run prediction.spam_message_predict(“New Message”)
