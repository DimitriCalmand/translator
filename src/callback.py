import tensorflow as tf
import keras


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, tokenizer, verbose = 10, model = None):
        """Displays a batch of outputs after every epoch"""
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.batch = batch
        if model is not None:
            self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self. verbose == 0:
            batch = next(self.batch)
            print()
            for i in range(4):
                source = batch[0]["encoder_inputs"].numpy()[i:i+1]
                target = batch[0]["decoder_inputs"].numpy()[i:i+1]
                predicted = self.model.generate(source, self.tokenizer)
                str_predicted = self.tokenizer.decode(predicted.numpy()[0])
                str_target = self.tokenizer.decode(target[0])
                str_source = self.tokenizer.decode(source[0])
                print('-'*80)
                print('source : ', str_source)
                print("target : ", str_target)
                print('Prediction: ', str_predicted)
            print('-'*80)
