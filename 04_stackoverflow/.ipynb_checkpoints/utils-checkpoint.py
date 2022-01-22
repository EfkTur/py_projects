import pandas as pd
from itertools import chain
from collections import Counter
from datetime import datetime
import re



def tags_cleaner(tags):
    """Removes '<' and '>' in each tag from sample. Then returns a list of tags from sample"""
    temp = re.sub("<","", tags)
    temp = re.sub(">"," ", temp)
    return temp.split()

def most_common_selector(tags_counter,list_to_check):
    if len(list_to_check) == 1:
        return list_to_check[0]
    tag_max = list_to_check[0]
    freq_max = tags_counter[list_to_check[0]]
    for tag in list_to_check[1:]:
        if tags_counter[tag]>freq_max:
            freq_max = tags_counter[tag]
            tag_max = tag
    return tag_max

def checker(value, iter):
    if value in iter:
        return True
    return False

def stop_words_check(word_list, stopwords):
    res_list = []
    for word in word_list: 
        if word not in stopwords:
            res_list.append(word)
    return res_list

