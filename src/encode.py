import pandas as pd
import numpy as np
import tensorflow as tf

def load(path:str):
    df = pd.read_csv(path)
    french = df["fr"]
    english = df["en"]
    return french.tolist(), english.tolist()
class Tokenizer:
    def __init__(self,
            max_word = 10000,
            special_char = ",.?!':;|"
            ):
        self.max_word = max_word 
        self.special_char = special_char
        self.config = {}
        self.recurent_word = {}
    @property
    def nb_word(self):
        return len(self.config)
    def train(self, input):
        for sentence in input:
            for char in self.special_char :
                sentence = sentence.replace(char, ' '+char)
            sentence = sentence.replace("  ", ' ')
            list_word = sentence.split(' ')
            for word in list_word:
                if word in self.recurent_word:
                    self.recurent_word[word] += 1
                else:
                    self.recurent_word[word] = 1
        sorted_keys = sorted(self.recurent_word,
                            key=lambda k:self.recurent_word[k])
        self.config["<oov>"] = 1
        for index,key in enumerate(reversed(sorted_keys)):
            if index >= self.max_word-1:
                break
            self.config[key] = index+2 
    def call(self, sentence):
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
        res = []
        for word in list_word:
            if word in self.config:
                res.append(self.config[word])
            else:
                res.append(self.config["<oov>"])
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
            {"encoder_inputs":inputs, "decoder_inputs":outputs[:, :-1]},
            {"outputs": outputs[:, 1:]}
            )
    inputs, outputs = data
    dataset = tf.data.Dataset.from_tensor_slices((list(inputs), list(outputs)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(transform)
    return dataset.shuffle(2000).prefetch(64).cache()
if __name__ == "__main__":
    path = "/home/dimitri/documents/python/artificial_inteligence/nlp/"+ \
            "translator/data/translate.csv"
    french, english = load(path)
    for_training = french + english
    tokenizer = Tokenizer()
    tokenizer.train(for_training)
    french_token = tokenizer.call(french)
    english_token = tokenizer.call(english)
    dataset = encode_using_tensorflow([french_token, english_token])
