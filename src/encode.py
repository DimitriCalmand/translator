import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from utils import *
import re
def remove_zeros(tensor):
    tensor = tf.boolean_mask(tensor, tf.not_equal(tensor, 0))
    return tensor
def pretty_print(liste):
    liste = liste.split(" ")
    res = ""
    for string in liste:
        if (string == START_WORD or string == END_WORD):
            continue
        res += string + " "
    return res
def decode(tensor, tokenizer):
    tensor = [remove_zeros(tensor).numpy()]
    liste = tokenizer.sequences_to_texts(tensor)
    return pretty_print(liste[0])

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
def encode(liste, language = 'fr'):
    encode_special_char(liste)
    tokenizer = tokenizer_fr
    if (language == 'en'):
        tokenizer = tokenizer_en
    res = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in liste]
    res = tokenizer.texts_to_sequences(res)
    res = pad_sequences(
                res, 
                maxlen=MAX_LENGHT, 
                padding='post', 
                truncating='post'
                )
    return res
