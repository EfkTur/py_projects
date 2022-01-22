import streamlit as st
import requests
import json
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
from stackoverflow_utils import stop_words_check
from deployment_utils import set_simplifier
from keras.preprocessing.sequence import pad_sequences



def input_preparation(user_input:str,tokenizer,maxlen):
    '''
    Raison d'être:
    Tokenize, remove stopwords and lemmatize, and join back in string to be put in a list.
    Then tranforms the sequence of token into a vector that we feed into the model
    ---------------------------------------------------------------
    Args: 
    user_input: Text entered by the user
    tokenizer: Tokenizer user in the neural network
    maxlen: Maximal number of token to consider
    ----------------------------------------------------------------
    Returns:
    results: A list of sequences
    '''
    results = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    #This line below tokenize, remove punct, and put all the words in lower cases
    transformed_input = text_to_word_sequence(str(user_input),filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=' ')
    #This line below removes the stopwords in the list of tokens
    transformed_input = stop_words_check(transformed_input,stop_words)
    #This line below does the lemmatization of the new token list
    transformed_input = [lemmatizer.lemmatize(token, 'v') for token in transformed_input]
    #This line below does remove the words not in our universe of words
    #transformed_input = tokenizer.texts_to_sequences(transformed_input)
    #transformed_input = tokenizer.sequences_to_texts(transformed_input)
    #Now we just join back the strings in a single string which we put in list
    transformed_input = ["".join(token) for token in transformed_input]
    #Then we want to create the feature vectors
    transformed_input = tokenizer.texts_to_sequences(transformed_input)
    data = pad_sequences(transformed_input, maxlen=maxlen)
    line = data.shape[0]
    for rank in range(line):
        results.append(list(data[rank]))
    return results


def request_prediction(model_uri,data,binarizer):
    '''
    Raison d'être:
    Makes an API call to the app to get the corresponding tags for the text input
    
    Args:
    model_uri: localhost adress where the API is live
    data: preprocessed input from the user. It is the output of input_preparation() function
    binarizer: Binarizer used in the training labels

    Returns:
    predicted_labels: set of the predicted_labels
    '''
    
    headers = {"Content-Type": "application/json"}

    data_json = '{"data":'+str(data)+'}'
    response = requests.request(
        method='POST', headers=headers, url=model_uri, data=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    parsed_response = json.loads(response.content)
    seuil = 0.05
    prediction = np.where(np.array(parsed_response) >= seuil, 1, 0)
    predicted_labels = binarizer.inverse_transform(prediction)
    predicted_labels = set_simplifier(predicted_labels)
    return predicted_labels


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    with open('./exports/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('./exports/mlb.pickle', 'rb') as handle:
        mlb = pickle.load(handle)


    st.title('Stackoverflow Tag Prediction')

    txt = st.text_area('Post to analyze', placeholder='Write your text here')
    
    predict_btn = st.button('Request Tags')
    


    if predict_btn:
        data = txt
        data = input_preparation(data,tokenizer=tokenizer,maxlen=180)
        #pred = input_preparation(data,tokenizer=tokenizer,maxlen=180)
        pred = request_prediction(MLFLOW_URI, data, mlb)
        st.write('Your predicted tags are: '+str(pred))

if __name__ == '__main__':
    main()
