import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from utils import *
import re
def remove_zeros(tensor):
    tensor = tf.boolean_mask(tensor, tf.not_equal(tensor, 0))
    return tensor
def pretty_print(l, list_oov):
    res_list = []
    i = 0
    for liste in l:
        liste = liste.split(" ")
        res = ""
        k = 0
        for string in liste:
            if (string == START_WORD or string == END_WORD):
                continue
            if list_oov is not None and string == '<oov>':
                res += list_oov[i][k] + ' '
                k += 1
                continue
            res += string + " "
        res_list.append(res)
        i += 1
    return res_list 
def decode(tensor, tokenizer, list_oov):
    tensor = [remove_zeros(tensor).numpy()]
    liste = tokenizer.sequences_to_texts(tensor)
    return pretty_print(liste, list_oov)

def encode_mail(liste):
    for i in range(len(liste)):
        spliter = liste[i].split(' ')
        for k in range(len(spliter)):
            if '@' in spliter[k]:
                spliter[k] = MAIL_WORD
        liste[i] = ' '.join(spliter)

def encode_mail(spliter, k):
    if '@' in spliter[k]:
        spliter[k] = MAIL_WORD
def encode_number(spliter, k):
    if re.search(r'\d', spliter[k]):
        spliter[k] = NUMBER_WORD 
def encode_name(spliter, k):
    if spliter[k][0].isupper():
        spliter[k] = NAME_WORD 
def encode_special_char(liste):
    start = True 
    for i in range(len(liste)):
        spliter = text_to_word_sequence(
                liste[i],
                filters='!"#$%&()*+,-/:;=?[\\]^_`{|}~\t\n',
                lower = False
                )
        for k in range(len(spliter)):
            if len(spliter[k]) != 0:
                save = spliter[k]
                encode_mail(spliter, k)
                encode_number(spliter, k)
                if not start:
                    #encode_name(spliter, k)
                    pass
                else:
                    start = False 
                if '.' in save:
                    start = True 
        liste[i] = ' '.join(spliter)
def encode(liste, list_oov = None, language = 'fr'):
    encode_special_char(liste)
    tokenizer = tokenizer_fr
    if (language == 'en'):
        tokenizer = tokenizer_en
    string = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in liste]
    res = tokenizer.texts_to_sequences(string)
    if list_oov is not None:
        for i in range(len(res)):
            tmp = [] 
            for k in range(len(res[i])):
                string_split = string[i].split(' ')
                if res[i][k] == 1:
                    tmp.append(string_split[k])
            list_oov.append(tmp)
        
                
    res = pad_sequences(
                res, 
                maxlen=MAX_LENGHT, 
                padding='post', 
                truncating='post'
                )
    return res
