#!/usr/bin/python3
from transformer import Transformer
import tensorflow as tf
from callback import DisplayOutputs
from time import time

from encode import *
CHECKPOINT_PATH = "training_1/cp.ckpt"
def training(get_checkpoint, returning = False):
    path = "/home/dimitri/documents/python/ia/nlp/"+ \
                "translator/data/translate.csv"
    french , english = load(path)
    for_training = french + english
    tokenizer = Tokenizer(max_word = 10000)
    tokenizer.train(for_training)
    french_token = tokenizer.call(french, padding = 0)
    english_token = tokenizer.call(english, padding = 0)
    dataset = encode_using_tensorflow([french_token, english_token], batch_size = 10)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing = 0.1
            )
    batch = next(iter(dataset))
    optimizer = tf.keras.optimizers.Adam()
    model = create_model(tokenizer.max_lenght, tokenizer.max_word, get_checkpoint = get_checkpoint)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                 filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=0)

    display_cb = DisplayOutputs(batch, tokenizer, verbose = 2, model = model)
    model.compile(optimizer = optimizer, loss = loss_fn)
    cur_time = time()
    model.fit(dataset, epochs = 10, verbose = 1, callbacks = [display_cb, cp_callback])
    print("time execution = ", time() - cur_time)
    #model.fit(dataset, epochs = 10)
def create_model(msl, vs, get_checkpoint : bool):
    model = Transformer(
            nb_encoder = 1,
            nb_decoder = 1,
            nb_heads = 1,
            embed_dim = 128,
            feed_forward_dim = 100,
            max_sequence_length = msl, 
            vocab_size = vs 
            )
    if (get_checkpoint):
        model.load_weights(CHECKPOINT_PATH)
    return model 
def test_model():
    model, tokenizer, callback = training(True, returning = True)
    model.fit(dataset, epochs = 1, verbose = 1, callbacks = callback)
    predicted = model.generate("Bonjour comment ca va", tokenizer)
    str_predicted = tokenizer.decode(predicted.numpy()[0])
    print(str_predicted)
#test_model()


training(True, True)

