import tensorflow as tf
import numpy as np
from encoder_layer import embedding_layer

class decoder_layer(tf.keras.layers.Layer):
    def __init__(self,
            num_heads,
            embed_dim,
            feed_forward_dim,
            dropout=0.0,
            ):
        """decoder layer"""
        super(decoder_layer, self).__init__()
        self.mask_multi_h_a = tf.keras.layers.MultiHeadAttention(
                num_heads,
                embed_dim,
                dropout=0.0)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_h_a = tf.keras.layers.MultiHeadAttention(
                num_heads,
                embed_dim,
                dropout=0.0)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                    tf.keras.layers.Dense(embed_dim)
                ]
            )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    def self_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    def call (self, inputs, encoder_input, training=False):
        mask = self.self_attention_mask(inputs)
        multi_mask_h_a = self.mask_multi_h_a(
                inputs,
                inputs,
                attention_mask=mask
                )
        dropout1 = self.dropout1(multi_mask_h_a, training=training)
        norm1 = self.norm1(dropout1 + inputs)
        multi_h_a = self.multi_h_a(
                norm1,
                encoder_input
                )
        dropout2 = self.dropout2(multi_h_a, training = training)
        norm2 = self.norm2(dropout2 + norm1)
        ffn = self.ffn(norm2)
        norm3 = self.norm3(ffn + norm2)
        return norm3

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,
            nb_decoder,
            num_heads,
            embed_dim,
            feed_forward_dim,
            max_sequence_length,
            vocab_size,
            dropout=0.0
            ):
        super(TransformerDecoder, self).__init__()
        self. embedding = embedding_layer(
                    max_sequence_length,
                    vocab_size,
                    embed_dim
                    )
        self.decoders = []
        for i in range(nb_decoder):
            self.decoders.append(
                    decoder_layer(
                        num_heads,
                        embed_dim,
                        feed_forward_dim,
                        dropout=dropout
                        )
                    )
    def decode(self, decoder_input, encoder_output, training = False):
        res = decoder_input 
        for decoder in self.decoders:
            res = decoder.call(res, encoder_output, training = training)
       # tf.print("decoder mean = ", tf.math.reduce_mean(tf.math.abs(res)))
       # tf.print("decoder res = ", res)
        return res

    def call (self, inputs,  training=False):
        decoder_input, encoder_output = inputs
        embed_decoder = self.embedding(decoder_input)

        return self.decode(embed_decoder, encoder_output, training)
