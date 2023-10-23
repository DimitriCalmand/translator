#!/usr/bin/python3
from transformer import Transformer
import tensorflow as tf
from callback import DisplayOutputs
from time import time

from encode import *
path = "/home/dimitri/documents/python/ia/nlp/"+ \
            "translator/data/translate.csv"
french , english = load(path)
for_training = french + english
tokenizer = Tokenizer(max_word = 6000)
tokenizer.train(for_training)
french_token = tokenizer.call(french, padding = 0)
english_token = tokenizer.call(english, padding = 0)
dataset = encode_using_tensorflow([french_token, english_token], batch_size = 10)

loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing = 0.1
        )
batch = next(iter(dataset))
optimizer = tf.keras.optimizers.Adam()
model = Transformer(
        nb_encoder = 4,
        nb_decoder = 1,
        nb_heads = 1,
        embed_dim = 16,
        feed_forward_dim = 100,
        max_sequence_length = tokenizer.max_lenght,
        vocab_size = tokenizer.max_word 
        )
display_cb = DisplayOutputs(batch, tokenizer, verbose = 99, model = model)
model.compile(optimizer = optimizer, loss = loss_fn)
cur_time = time()
model.fit(dataset, epochs = 100, verbose = 0, callbacks = [display_cb])
print("time execution = ", time() - cur_time)
#model.fit(dataset, epochs = 10)

