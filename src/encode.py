import pandas as pd
import numpy as np
from tensorflow.data import Dataset 
from sklearn.model_selection import train_test_split
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
            get_file = False 
            ):
        self.special_char = special_char
        self.saver_config = 'saver/saver_config.json'
        self.saver_properties = 'saver/saver_properties.json'
        if not get_file :
            self.config = {}
            self.max_lenght = 0
            self.max_word = max_word 
        else:
            self.get()
        self.recurent_word = {}
        self.start_token = start_token
        self.end_token = end_token

    @property
    def nb_word(self):
        return len(self.config)

    def train(self, input):
        self.config["<oov>"] = 1
        self.config[self.start_token] = 2
        self.config[self.end_token] = 3
        for sentence in input:
            for char in self.special_char :
                sentence = sentence.replace(char, ' '+char)
            sentence = sentence.replace("  ", ' ')
            list_word = sentence.split(' ')
            prev = self.max_lenght
            self.max_lenght = max(self.max_lenght, len(list_word) + 2)
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
        self.max_word = len(self.config) + 1
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
                "max_word" : self.max_word,
                "max_lenght" : self.max_lenght
                }
            json.dump(properties, f)

    def get(self):
        with open(self.saver_config, 'r') as f:
            self.config = json.load(f)

        with open(self.saver_properties, 'r') as f:
            properties = json.load(f)
            self.max_word = properties["max_word"]
            self.max_lenght = properties["max_lenght"]
        
def encode_using_hugging():
    """ man page
    https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config """

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenize_output = tokenizer(word)["input_ids"]
    return tokenize_output

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
    print(french_train)
    print("autre = ")
    print(french_test)
    
    french_token_train = tokenizer.call(french_train, 0)
    english_token_train = tokenizer.call(english_train, 0)
    french_token_test = tokenizer.call(french_test, 0)
    english_token_test = tokenizer.call(english_test, 0)
    train = [french_token_train, english_token_train]
    test = [french_token_test, english_token_test]
    return tokenizer, train, test
