import pandas as pd
import numpy as np
from tensorflow.data import Dataset 

def load(path:str):
    df = pd.read_csv(path)
    french = df["fr"]
    english = df["en"]
    return french.tolist(), english.tolist()
class Tokenizer:
    def __init__(self,
            max_word = 10000,
            special_char = ",.?!':;|",
            start_token = '<',
            end_token = '>'
            ):
        self.max_word = max_word 
        self.special_char = special_char
        self.config = {}
        self.recurent_word = {}
        self.start_token = start_token
        self.end_token = end_token
        self.max_lenght = 0

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

def main():
    path = "/home/dimitri/documents/python/ia/nlp/"+ \
            "translator/data/translate.csv"
    french, english = load(path)
    for_training = french + english
    tokenizer = Tokenizer()
    tokenizer.train(for_training)
    french_token = tokenizer.call(french, 0)
    english_token = tokenizer.call(english, 0)
    dataset = encode_using_tensorflow([french_token, english_token])
if __name__ == "__main__":
    main()

