from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from AttentionedNetworks.Keras_many_to_one_attention.attention.attention import attention_3d_block


def attention_lstm(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    x = attention_3d_block(x)
    # x = Dropout(0.5)(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    return model
