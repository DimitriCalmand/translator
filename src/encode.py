import pandas as pd
import numpy as np
from tensorflow.data import Dataset 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import json

def load(path:str):
    df = pd.read_csv(path)
    french = df["fr"][:100]
    english = df["en"][:100]
    return french.tolist(), english.tolist()
class Tokenizer:
    def __init__(self,
            max_word = 10000,
            special_char = ",.?!':;|",
            start_token = '<',
            end_token = '>',
            get_file = False,
            max_lenght = 100 
            ):

        self.special_char = special_char
        self.saver_config = 'saver/saver_config.json'
        self.saver_properties = 'saver/saver_properties.json'
        if not get_file :
            self.config = {}
            self.max_lenght = max_lenght 
            self.cur_max_word = max_word 
            self.max_word = max_word
        else:
            self.get()
        
        self.recurent_word = {}
        self.start_token = start_token
        self.end_token = end_token
        self.mean = 0
        self.nb = 0
        self.w = 0
    @property
    def nb_word(self):
        return len(self.config)

    def train(self, input):
        self.config["<oov>"] = 1
        self.config[self.start_token] = 2
        self.config[self.end_token] = 3
        for sentence in input:
            if isinstance(sentence, float):
                self.w += 1
                continue
            for char in self.special_char :
                sentence = sentence.replace(char, ' '+char)
            sentence = sentence.replace("  ", ' ')
            list_word = sentence.split(' ')
            if (len(list_word) + 2 > self.max_lenght):
                self.w += 1
                continue
            prev = self.max_lenght
            self.mean += len(list_word)
            self.nb += 1
            #self.max_lenght = max(self.max_lenght, len(list_word) + 2)
            for word in list_word:
                if word in self.recurent_word:
                    self.recurent_word[word] += 1
                else:
                    self.recurent_word[word] = 1
        sorted_keys = sorted(self.recurent_word,
                            key=lambda k:self.recurent_word[k])
        for index,key in enumerate(reversed(sorted_keys)):
            if index >= self.max_word-1:
                break
            self.config[key] = index+4 
        self.cur_max_word = len(self.config) + 1
    def call(self, sentence, padding=0):
        if isinstance(sentence, tuple):
            return (self.call(sentence[0], self.call(sentence[1])))
        if isinstance(sentence, list):
            res = []
            for tmp_sentence in sentence:
                res.append(self.call(tmp_sentence))
            return res
        for char in self.special_char :
            sentence = sentence.replace(char, ' '+char)

        sentence = sentence.replace("  ", ' ')
        list_word = sentence.split(' ')
        res = [self.config[self.start_token]]
        for word in list_word:
            if word in self.config:
                res.append(self.config[word])
            else:
                res.append(self.config["<oov>"])
        res.append(self.config[self.end_token])
        if padding is not None:
            prev_res = len(res)
            res += [0]*(self.max_lenght -len(res))
        return res
    def decode(self, inputs):
        res = ""
        keys = list(self.config.keys())
        values = list(self.config.values())
        for token in inputs:
            if (token == 0):
                continue
            res += keys[values.index(token)] 
            res += " "
        return res

    def save(self):
        with open(self.saver_config, 'w') as f:
            json.dump(self.config, f)

        with open(self.saver_properties, 'w') as f: 
            properties = {
                "cur_max_word" : self.max_word,
                "max_lenght" : self.max_lenght
                }
            json.dump(properties, f)

    def get(self):
        with open(self.saver_config, 'r') as f:
            self.config = json.load(f)

        with open(self.saver_properties, 'r') as f:
            properties = json.load(f)
            self.cur_max_word = properties["cur_max_word"]
            self.max_lenght = properties["max_lenght"]
        
def encode_using_tensorflow(data:list, batch_size:int=64):
    def transform(inputs, outputs):
        return (
                {"encoder_inputs":inputs[:, 1:], "decoder_inputs":outputs[:, :-1]},
            {"outputs": outputs[:, 1:]}
            )
    inputs, outputs = data
    dataset = Dataset.from_tensor_slices((inputs, outputs))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(transform)
    return dataset.shuffle(2000).prefetch(9).cache()
def get_data(ratio = 0.2, get_file = False):
    path = "/home/dimitri/documents/python/ia/nlp/"+ \
            "translator/data/translate.csv"
    french, english = load(path)
    tokenizer = Tokenizer(get_file = get_file)
    if not get_file :
        for_training = french + english
        tokenizer.train(for_training)

    french_train, french_test, english_train, english_test = train_test_split(
            french, english, test_size = ratio)

    french_token_train = tokenizer.call(french_train, 0)
    english_token_train = tokenizer.call(english_train, 0)
    french_token_test = tokenizer.call(french_test, 0)
    english_token_test = tokenizer.call(english_test, 0)
    train = [french_token_train, english_token_train]
    test = [french_token_test, english_token_test]
    return tokenizer, train, test
def apply_tokenizer(tokenizer, x):
    english_train = x['en']
    french_train = x['fr']
    inputs = tokenizer.call(french_train, 0)
    outputs = tokenizer.call(english_train, 0)
    res = (
            {"encoder_inputs":inputs[:, 1:], "decoder_inputs":outputs[:, :-1]},
            {"outputs": outputs[:, 1:]}
            )
    return res

def main():
    file = '../data/translate.csv'
    data = tf.data.experimental.make_csv_dataset(
            file,
            9, #batch_size
            column_names = ['en', 'fr'],
            shuffle = False,
            )
    
    data = data.map(lambda x: apply_tokenizer(tokenizer, x))
    for i in data:
        print(i)
        break
