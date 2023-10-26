#!/usr/bin/python3
from transformer import Transformer
import tensorflow as tf
from callback import DisplayOutputs
from time import time
#from tensorflow_encode import load
from encode import *
from dataset import * 

CHECKPOINT_PATH = "training_1/cp.ckpt"

def training(get_checkpoint, get_tokenizer = False, stop_early = False):
    tokenizer = Tokenizer(get_file = get_tokenizer)
    dataset = CustomDataset("../data/translate.csv", batch_size = 10)
    _, train, test = get_data(get_file = get_tokenizer) 
    dataset1 = encode_using_tensorflow(train, batch_size = 3)
    print(dataset1)
    print(dataset)
    for i in dataset:
        print(i)
        break
    return None, None 
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing = 0.1
            )
    #batch_test = iter(ds_test)
    optimizer = tf.keras.optimizers.Adam()
    model = create_model(
            tokenizer.max_lenght,
            tokenizer.max_word,
            get_checkpoint = get_checkpoint)
    model.tokenizer = tokenizer
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #                                             filepath=CHECKPOINT_PATH,
    #                                             save_weights_only=True,
    #                                             verbose=0)

    #display_cb = DisplayOutputs(batch_test, tokenizer, verbose = 10, model = model)
    model.compile(optimizer = optimizer, loss = loss_fn)
    cur_time = time()
    if stop_early:
        return model, tokenizer
    model.fit(dataset, epochs = 2, verbose = 1) 
    print("time execution = ", time() - cur_time)
    #tokenizer.save()
    return None, None
    #model.fit(ds_train, epochs = 10)
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
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore(CHECKPOINT_PATH).expect_partial()

   #     model.load_weights(CHECKPOINT_PATH)
    return model 
def main():
    stop_early = False 
    model, tokenizer = training(False, get_tokenizer = True, stop_early = stop_early)
    if stop_early : 
        string = "en Ontario ."
        print(model.predict_str(string , tokenizer))
        print("expected : 'Observatories Ontario 's Sudbury Neutrino Observatory is established .'")
        


