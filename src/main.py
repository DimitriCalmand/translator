from transformer import Transformer
import tensorflow as tf
from callback import * 
from time import time
from utils import *
from encode import *
from dataset import * 
from tensorflow.keras.callbacks import TensorBoard

CHECKPOINT_PATH = "training_1/cp.ckpt"

def training(get_checkpoint, stop_early = False):
    ds_train = CustomDataset("../data/train.h5", batch_size = BATCH_SIZE)
    ds_test = CustomDataset("../data/test.h5", batch_size = 4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing = 0.1
            )
    batch_test = iter(ds_test)
    optimizer = tf.keras.optimizers.Adam()
    model = create_model(get_checkpoint = get_checkpoint)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #                                             filepath=CHECKPOINT_PATH,
    #                                             save_weights_only=True,
    #                                             verbose=0)

# Création du callback TensorBoard avec la surveillance de la mémoire
    tensorboard_callback = TensorBoard(
            log_dir="./logs", 
            histogram_freq=1
            )
    display_cb = DisplayOutputs(batch_test, verbose = 20, model = model)
    model.compile(optimizer = optimizer, loss = loss_fn)
    cur_time = time()
    if stop_early:
        return model
    model.fit(
            ds_train,
            epochs = 100,
            verbose = 1,
            callbacks = [display_cb, tensorboard_callback]
            ) 
    print("time execution = ", time() - cur_time)
    return None, None
def create_model(get_checkpoint : bool):
    model = Transformer(
            nb_encoder = 1,
            nb_decoder = 2,
            nb_heads = 1,
            embed_dim = 128, 
            feed_forward_dim = 100,
            max_sequence_length = MAX_LENGHT, 
            vocab_size = NUM_WORDS 
            )
    if (get_checkpoint):
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore(CHECKPOINT_PATH).expect_partial()
    return model 

def main():
    stop_early = False 
    model = training(False, stop_early = stop_early)
    if stop_early : 
        string = "en Ontario ."
        print(model.predict_str(string))
        print("expected : 'Observatories Ontario 's Sudbury Neutrino Observatory is established .'")
