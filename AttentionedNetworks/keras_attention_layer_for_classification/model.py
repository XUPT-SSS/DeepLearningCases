from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from AttentionedNetworks.keras_attention_layer_for_classification.attention_layer import AttentionWithContext


def attention_lstm(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    x = AttentionWithContext()(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    return model

def attention_lstm_expliable(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    attwithcontext = AttentionWithContext()
    x = attwithcontext(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    fn = K.function([inp], [attwithcontext.att_weights])
    return model, fn

def lstm(hidden_size, max_len, max_features, v_size):
    # Input for variable-length sequences of integers
    encoder_inputs = Input(shape=(None,), dtype="int32", name='inputs')
    # Embed each integer in a 128-dimensional vector
    x = Embedding(max_features, v_size, name='embedding_layer')(encoder_inputs)
    # Add 2 bidirectional LSTMs
    encoder_lstm = GRU(hidden_size, name='encoder_lstm')
    encoder_state = encoder_lstm(x)

    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(encoder_state)
    model = Model(encoder_inputs, outputs)
    model.summary()
    return model
