import pandas as pd
import re
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def tags_cleaner(tags):
    """
    Raison d'être:
        Removes '<' and '>' in each tag from sample. 
    Args: 
        Tags to be deleted
    Returns: 
        List of cleaned tags
    """
    temp = re.sub("<","", tags)
    temp = re.sub(">"," ", temp)
    return temp.split()

def most_common_selector(tags_counter,list_to_check):
    """
    Raison d'être:
        Selects the most common tags in our database
    Args: 
        The counter, the list to check for tag
    Returns: 
        A list of common tags
    """
    res = []
    for tag in list_to_check:
        if tag in tags_counter.keys():
            res.append(tag)
    return res

def checker(value, iter):
    """
    Raison d'être: 
        Checks if an element is in an iterable
    Args: 
        Value to be checked
    Returns: 
        Boolean
    """
    if value in iter:
        return True
    return False

def stop_words_check(word_list, stopwords):
    """
    Raison d'être:
        Checks for stopwords in a list of tokens. Note that this will also remove the words with short length
    Args: 
        Word_list a list of word, Stopwords to be checked
    Returns: 
        A list of tokens without stop words
    """
    res_list = []
    for word in word_list: 
        if word not in stopwords and len(word)>2:
            res_list.append(word)
    return res_list

def tokenizer(sentence):
    """
    Raison d'être: 
        Function to use keras' tokenizer. Also remove punctuation, and puts everything in lower case. 
    Args: 
        Sentence to be checked 
    Returns: 
        A list of tokens
    """
    res = tf.keras.preprocessing.text.text_to_word_sequence(sentence,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=' ')
    return res

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def clean_punct(text, top_tags_counter): 
    """
    Raison d'être:
        Clean punctuation in a given sequence of text 

    Args:
        text: text to be cleaned
        top_tags_counter:

    Returns:    
    """
    punct = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789'
    words = tf.keras.preprocessing.text.text_to_word_sequence(text,filters=punct,lower=True)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    #remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in top_tags_counter.keys():
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

def clean_text(text):
    '''
    Raison d'être:
        Takes a raw text and makes it in usable format

    Args:
        text: a text to be cleaned

    Returns:
        text: a cleaned text    
    '''
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def predict_unsupervised_tags(text,lda_model,id2word):
    """
    Raison d'être:
        Predict tags of a preprocessed text
    
    Args:
        text(list): preprocessed text
        
    Returns:
        relevant_tags(list): list of tags
    """
    
    corpus_new = id2word.doc2bow(text)
    topics = lda_model.get_document_topics(corpus_new)
    
    #find most relevant topic according to probability
    relevant_topic = topics[0][0]
    relevant_topic_prob = topics[0][1]
    
    for i in range(len(topics)):
        if topics[i][1] > relevant_topic_prob:
            relevant_topic = topics[i][0]
            relevant_topic_prob = topics[i][1]
            
    #retrieve associated to topic tags present in submitted text
    potential_tags = lda_model.get_topic_terms(topicid=relevant_topic, topn=20)
    
    relevant_tags = [id2word[tag[0]] for tag in potential_tags if id2word[tag[0]] in text]
    
    return relevant_tags


def scorer_jaccard(y_true,y_pred):
    '''
    Raison d'être:
        Making jaccard_score a usable metric in the NN model

    Args:
    y_true: Actual labels
    y_pred: Predicted Labels

    Returns:
    A scalar which represent the jaccard_score
    '''
    return jaccard_score(y_true,y_pred,average='weighted')

def metrics_score(model, df, y_true, y_pred):
    """
    Raison d'être:
        Compilation function of metrics specific to multi-label
        classification problems in a Pandas DataFrame.
        This dataFrame will have 1 row per metric
        and 1 column per model tested. 

    Args:
        model : string
            Name of the tested model
        df : DataFrame 
            DataFrame to extend. 
            If None : Create DataFrame.
        y_true : array
            Array of true values to test
        y_pred : array
            Array of predicted values to test
    
    Returns:
        temp_df: a pandas DataFrame including the scores of chosen metrics
    """
    if(df is not None):
        temp_df = df
    else:
        temp_df = pd.DataFrame(index=["Accuracy", "F1",
                                      "Jaccard", "Recall",
                                      "Precision"],
                               columns=[model])
        
    scores = []
    scores.append(accuracy_score(y_true, y_pred))
    scores.append(f1_score(y_pred, y_true, average='weighted'))
    scores.append(jaccard_score(y_true, y_pred, average='weighted'))
    scores.append(recall_score(y_true, y_pred, average='weighted'))
    scores.append(precision_score(y_true, y_pred, average='weighted'))
    temp_df[model] = scores
    
    return temp_df

def set_simplifier(set_values):
    '''
    Raison d'être: 
        Simplifies the structure of a multilevel set
    Args: 
        Set_values, a set to simplify

    Returns: 
        new_set, a new simplified set 
    '''
    new_set = set()
    for value in set_values:
        for entry in value:
            new_set.add(entry)
    return new_set

def NN_model_results(modelNN,y_test_embedding,X_test_embedding,mlb,seuil):
    '''
    Raison d'être:
        Putting the predicted labels of our labels in a readable and usable format

    Args:
        modelNN: the used NN model 
        y_test_embedding: the prepared y_test dataset
        X_test_embedding: the prepared X_test dataset
        mlb: MultiLabelBinarizer used in the data preparation
        seuil: A float used as a threshold to determine which of the predicted labels will pass the cut

    Returns:
        results: A pandas DataFrame that includes the comparison between actual labels and predicted labels
    ''' 

    results = pd.DataFrame()
    for i in range(len(y_test_embedding)):
        prediction = modelNN.predict(np.array([X_test_embedding[i]]))
        prediction = np.where(prediction >= seuil, 1, 0)
        predicted_labels = mlb.inverse_transform(prediction)
        actual_labels = mlb.inverse_transform(np.array([y_test_embedding[i]]))
        results = results.append({
        'Actual_Labels': set_simplifier(actual_labels),
        'Predicted_Labels': set_simplifier(predicted_labels),
        },ignore_index=True)
    
    return results

def threshold_optimizer(possible_thresholds,modelNN,y_val_embed,X_val_embed,mlb):
    '''
    Raison d'être:
        Determining the optimal threshold that mamiximises jaccard_score. Note that to avoid major overfitting, 
        we use a validation set to run this function. We think this is best practice. 

    Args: 
        possible_threshold: An iterable of threshold to be tested
        modelNN: the NN model used
        y_val_embed: the validation set of y values
        X_val_embed: the validation set of X values
        mlb: MultiLabelBinarizer used to train the model

    Returns:
        optimal_threshold: a scalar that represents the optimal_threshold to be used in the NN_model_results() function
    '''

    optimal_threshold = possible_thresholds[0]
    jaccard_max = 0
    jaccard_list = []
    
    resultats = NN_model_results(modelNN,y_val_embed,X_val_embed,mlb,optimal_threshold)
    true_encoded = mlb.transform(resultats.Actual_Labels)
    pred_encoded = mlb.transform(resultats.Predicted_Labels)
    jaccard_max = jaccard_score(true_encoded,pred_encoded,average='weighted')
    jaccard_list.append(jaccard_max)
    
    for value in possible_thresholds[1:]:
        resultats = NN_model_results(modelNN,y_val_embed,X_val_embed,mlb,value)
        true_encoded = mlb.transform(resultats.Actual_Labels)
        pred_encoded = mlb.transform(resultats.Predicted_Labels)
        temp = jaccard_score(true_encoded,pred_encoded,average='weighted')
        jaccard_list.append(temp)
        if temp > jaccard_max:
            jaccard_max = temp
        optimal_threshold = value
    return optimal_threshold


def get_features(text_series,tokenizer,maxlen):
    """
    Raison d'être:
        Transforms text data to feature_vectors that can be used in the ml model.
    
    Args:
        text_series: An iterable with text to be transformed
        tokenizer: Tokenizer to be used to transform the text series
        maxlen: Maximum number of words authorized in each of the text series
    
    Returns:
        A padded sequence
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)

def display_scree_plot(pca):
    '''
    Raison d'être: 
        Displaying the plot of various cumulated variance in order to determine the optimal number of compenents to use

    Args:
        pca: A reductor. Note that although we name this 'pca', we can also use an SVD reductor

    Returns:
        None. Shows plt plot. 
    '''   
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
